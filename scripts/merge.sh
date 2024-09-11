CUDA_VISIBLE_DEVICES=0 python src/export_model.py \
    --model_name_or_path $MODEL \
    --adapter_name_or_path $ADAPTER \
    --template  $TEMPLATE \
    --finetuning_type lora \
    --resize_vocab True\
    --export_dir $EXPORT \
    --export_size 2 \
    --export_legacy_format False