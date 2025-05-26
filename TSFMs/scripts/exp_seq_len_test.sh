export CUDA_VISIBLE_DEVICES=1

python -u test.py --config-file ./configs/energy_data.json --seq_len 96 --model_save_path ./checkpoints/MixBEATS_Seq_96 --result_path ./results/MixBEATS_Seq_96

python -u test.py --config-file ./configs/energy_data.json --seq_len 168 --model_save_path ./checkpoints/MixBEATS_Seq_168 --result_path ./results/MixBEATS_Seq_168

python -u test.py --config-file ./configs/energy_data.json --seq_len 336 --model_save_path ./checkpoints/MixBEATS_Seq_336 --result_path ./results/MixBEATS_Seq_336

python -u test.py --config-file ./configs/energy_data.json --seq_len 720 --model_save_path ./checkpoints/MixBEATS_Seq_720 --result_path ./results/MixBEATS_Seq_720