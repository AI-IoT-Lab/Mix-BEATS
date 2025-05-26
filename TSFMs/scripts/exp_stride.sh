export CUDA_VISIBLE_DEVICES=1

python -u train.py --config-file ./configs/energy_data.json --stride 24 --model_save_path ./checkpoints/MixBEATS_Stride_24

python -u train.py --config-file ./configs/energy_data.json --stride 12 --model_save_path ./checkpoints/MixBEATS_Stride_12

python -u train.py --config-file ./configs/energy_data.json --stride 8 --model_save_path ./checkpoints/MixBEATS_Stride_8 --num_epochs 10

python -u train.py --config-file ./configs/energy_data.json --stride 4 --model_save_path ./checkpoints/MixBEATS_Stride_4 --num_epochs 10

python -u train.py --config-file ./configs/energy_data.json --stride 2 --model_save_path ./checkpoints/MixBEATS_Stride_2 --num_epochs 10 

python -u train.py --config-file ./configs/energy_data.json --stride 1 --model_save_path ./checkpoints/MixBEATS_Stride_1 --num_epochs 5 