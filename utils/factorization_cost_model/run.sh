echo "Training on GPU $1"
CUDA_VISIBLE_DEVICES=$1 python train_lstm.py
