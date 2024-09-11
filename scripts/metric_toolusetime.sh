export HF_ENDPOINT=https://hf-mirror.com
export CUDA_VISIBLE_DEVICES=0

python metric/knwoledge_time.py --data_path $DATA_PATH
