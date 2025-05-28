DEEPSPEED_CMD="deepspeed --num_gpus=4"
SCRIPT_PATH="src/continual_pretraining.py"

PARAM_GROUPS=(
    """
    --repetition_ratio 0.3 \
    --bin_extra_capacity 50 \
    --model_name meta-llama/Llama-3.2-1B \
    --input_file data/toy_train.txt \
    --validation_file data/toy_validation.txt \
    --output_dir models/Llama-3.2-1B_sp \
    --sequence_length 512 \
    --data_preprocess_type sp
    """
)

for PARAMS in "${PARAM_GROUPS[@]}"; do
    PARAMS=$(echo "$PARAMS" | tr -d '\n')
    echo "Running with parameters: $PARAMS"
    CUDA_VISIBLE_DEVICES=0,1,2,3 $DEEPSPEED_CMD $SCRIPT_PATH $PARAMS
done
