export CUDA_VISIBLE_DEVICES=1

python -u test.py --config-file ./configs/energy_data.json --stride 24 --model_save_path ./checkpoints/MixBEATS_Stride_24 --result_path ./results/MixBEATS_Stride_24

python -u test.py --config-file ./configs/energy_data.json --stride 12 --model_save_path ./checkpoints/MixBEATS_Stride_12 --result_path ./results/MixBEATS_Stride_12

python -u test.py --config-file ./configs/energy_data.json --stride 8 --model_save_path ./checkpoints/MixBEATS_Stride_8 --result_path ./results/MixBEATS_Stride_8

python -u test.py --config-file ./configs/energy_data.json --stride 4 --model_save_path ./checkpoints/MixBEATS_Stride_4 --result_path ./results/MixBEATS_Stride_4

python -u test.py --config-file ./configs/energy_data.json --stride 2 --model_save_path ./checkpoints/MixBEATS_Stride_2 --result_path ./results/MixBEATS_Stride_2

python -u test.py --config-file ./configs/energy_data.json --stride 1 --model_save_path ./checkpoints/MixBEATS_Stride_1 --result_path ./results/MixBEATS_Stride_1