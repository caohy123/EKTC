CUDA_VISIBLE_DEVICES=0 nohup python src/llmtuner_app.py \
    --model_name_or_path $BASE_MODEL \
    --template $TEMPLATE \
    --temperature 0.1 \
    --resize_vocab TRUE\
    --max_new_tokens 300 \
    --infer_backend huggingface \
    --do_sample