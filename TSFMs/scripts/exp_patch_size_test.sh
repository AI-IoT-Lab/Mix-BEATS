export CUDA_VISIBLE_DEVICES=0

python -u test.py --config-file ./configs/energy_data.json --patch_size 4 --model_save_path ./checkpoints/MixBEATS_Patch_4 --result_path ./results/MixBEATS_Patch_4

python -u test.py --config-file ./configs/energy_data.json --patch_size 8 --model_save_path ./checkpoints/MixBEATS_Patch_8 --result_path ./results/MixBEATS_Patch_8

python -u test.py --config-file ./configs/energy_data.json --patch_size 12 --model_save_path ./checkpoints/MixBEATS_Patch_12 --result_path ./results/MixBEATS_Patch_12

python -u test.py --config-file ./configs/energy_data.json --patch_size 24 --model_save_path ./checkpoints/MixBEATS_Patch_24 --result_path ./results/MixBEATS_Patch_24