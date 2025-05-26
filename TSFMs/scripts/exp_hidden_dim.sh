export CUDA_VISIBLE_DEVICES=0

python -u train.py --config-file ./configs/energy_data.json --hidden_dim 128 --model_save_path ./checkpoints/MixBEATS_Hidden_128

python -u train.py --config-file ./configs/energy_data.json --hidden_dim 256 --model_save_path ./checkpoints/MixBEATS_Hidden_256

python -u train.py --config-file ./configs/energy_data.json --hidden_dim 512 --model_save_path ./checkpoints/MixBEATS_Hidden_512

python -u train.py --config-file ./configs/energy_data.json --hidden_dim 1024 --model_save_path ./checkpoints/MixBEATS_Hidden_1024