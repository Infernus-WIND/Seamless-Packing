import torch
from tqdm import tqdm
from transformers import TrainerCallback
import json 
import math
import os


local_rank = int(os.environ.get("LOCAL_RANK", 0))


class ChunkedDataset(torch.utils.data.Dataset):
    def __init__(self, chunks, tokenizer, max_length=None):
        self.chunks = chunks
        self.tokenizer = tokenizer
        self.max_length = max_length or tokenizer.model_max_length
        self.total_padding_count = 0
        
        for chunk in self.chunks:
            pad_length = max(0, self.max_length - len(chunk))
            self.total_padding_count += pad_length

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        chunk = chunk[:self.max_length]
        pad_length = self.max_length - len(chunk)

        input_ids = torch.cat(
            [chunk, torch.full((pad_length,), self.tokenizer.pad_token_id, dtype=torch.long)]
        )
        labels = torch.cat(
            [chunk, torch.full((pad_length,), -100, dtype=torch.long)]
        )
        attention_mask = torch.cat(
            [torch.ones(len(chunk), dtype=torch.long), torch.zeros(pad_length, dtype=torch.long)]
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    def get_total_padding_count(self):
        return self.total_padding_count


def tokenize_and_chunk_by_seamless_packing(texts, tokenizer, sequence_length, repetition=0.3, bin_extra_capacity=50, padding=False):
    # Store the tokenized chunks
    chunks = []
    short_texts = []  # List for sequences shorter than sequence_length
    
    # Calculate the maximum overlap size
    max_overlap_size = int(sequence_length * repetition)

    # Process each text individually
    for text in tqdm(texts, desc="Processing texts"):
        tokenized = tokenizer(text.strip(), return_tensors="pt", truncation=False)["input_ids"].squeeze(0)
        length = len(tokenized)

        if length < sequence_length:
            # Text too short to fill a single chunk, defer for later
            short_texts.append(tokenized)
        else:
            # Calculate the number of full chunks that can fit into the text
            n = length // sequence_length

            if length % sequence_length == 0:
                # If the text length is exactly divisible by sequence_length
                for index in range(n):
                    chunks.append(tokenized[index * sequence_length: (index + 1) * sequence_length])
            else:
                if n * sequence_length < length < (sequence_length - max_overlap_size) * (n + 1) + max_overlap_size:
                    # Add full chunks
                    for index in range(n):
                        chunks.append(tokenized[index * sequence_length: (index + 1) * sequence_length])

                    # Add the remaining segment to short_texts
                    short_texts.append(tokenized[n * sequence_length:])
                else:
                    # Apply sliding window logic
                    overlap_length = int((sequence_length * (n + 1) - length) / (n + 1)) + 1

                    for index in range(n):
                        start = index * (sequence_length - overlap_length)
                        end = start + sequence_length
                        chunks.append(tokenized[start:end])

                    # Add the final chunk
                    chunks.append(tokenized[-sequence_length:])
    
    if padding:
        processed_short_texts = first_fit_decreasing(short_texts, sequence_length, sequence_length, padding=True)
    else:
        processed_short_texts = first_fit_decreasing(short_texts, sequence_length, sequence_length + bin_extra_capacity, padding=False)

    chunks += processed_short_texts
    return chunks


def first_fit_decreasing(tensor_list, sequence_length, bin_capacity, padding=False):
    # Step 1: Calculate the lengths of the tensors
    lengths = [t.numel() for t in tensor_list]  # Get the length (size) of each tensor
    
    # Step 2: Sort tensors by length in descending order
    sorted_indices = sorted(range(len(lengths)), key=lambda i: lengths[i], reverse=True)
    sorted_tensors = [tensor_list[i] for i in sorted_indices]
    
    # Step 3: Initialize bins (each bin is a list of tensors)
    bins = []
    bin_remaining_space = []  # Keeps track of remaining space in each bin
    
    # Step 4: Place tensors into bins using the FFD strategy
    for tensor in sorted_tensors:
        tensor_length = tensor.numel()
        placed = False
        
        # Try to place the tensor in the first bin that has enough space
        for i in range(len(bins)):
            if bin_remaining_space[i] >= tensor_length:
                bins[i].append(tensor)
                bin_remaining_space[i] -= tensor_length
                placed = True
                break
        
        # If it doesn't fit into any existing bin, create a new bin
        if not placed:
            bins.append([tensor])
            bin_remaining_space.append(bin_capacity - tensor_length)
    
    results = process_bins(bins, sequence_length, padding)
    return results


def process_bins(bins, threshold, padding=False):
    # Step 1: Merge tensors in each bin
    merged_bins = [torch.cat(bin_, dim=0) for bin_ in bins]  # Merge tensors within each bin
    
    # Step 2: Separate tensors into two categories
    long_tensors = []  # Tensors longer than the threshold
    short_tensors = []  # Tensors shorter than the threshold
    total_dropping_token = 0
    
    for tensor in merged_bins:
        if tensor.numel() > threshold:
            long_tensors.append(tensor[:threshold])  # Truncate to the threshold
            total_dropping_token += tensor.numel() - threshold
        else:
            if padding:
                long_tensors.append(tensor)
            else:
                short_tensors.append(tensor)  # Keep as-is
        
    # Step 3: Merge all short tensors into one long tensor
    if short_tensors:
        combined_short_tensor = torch.cat(short_tensors, dim=0)
    else:
        combined_short_tensor = torch.tensor([], dtype=torch.float32)
    
    # Step 4: Split the combined short tensor into fixed-length pieces
    fixed_length_tensors = []
    num_full_pieces = combined_short_tensor.numel() // threshold  # Number of full pieces
    for i in range(num_full_pieces):
        fixed_length_tensors.append(combined_short_tensor[i * threshold : (i + 1) * threshold])
    
    # Step 5: Combine all processed tensors (long tensors + fixed-length tensors from short)
    result_tensors = long_tensors + fixed_length_tensors    
    return result_tensors


class SaveMetricsCallback(TrainerCallback):
    def __init__(self, output_dir, training_args):
        self.output_dir = output_dir
        self.training_args = training_args
        self.metrics_file = f"{output_dir}/training_metrics.txt"
        self.params_file = f"{output_dir}/training_params.json"
        self.init_files()

    def init_files(self):
        if local_rank == 0:
            with open(self.metrics_file, "w") as f:
                f.write("Epoch\tMetrics\n")
            with open(self.params_file, "w") as f:
                json.dump(self._serialize_training_args(self.training_args), f, indent=4)

    def _serialize_training_args(self, training_args):
        serialized_args = {}
        for key, value in vars(training_args).items():
            try:
                json.dumps(value)
                serialized_args[key] = value
            except TypeError:
                serialized_args[key] = str(value)
        return serialized_args


    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and local_rank == 0:
            if "eval_loss" in metrics:
                metrics["perplexity"] = math.exp(metrics["eval_loss"])
            epoch = state.epoch if state.epoch is not None else metrics.get("epoch", "Unknown")
            with open(self.metrics_file, "a") as f:
                f.write(f"{epoch}\t{metrics}\n")
