export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0


python /metric/response_metrics.py --data_path $DATA_PATH
