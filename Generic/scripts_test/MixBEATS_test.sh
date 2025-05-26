export CUDA_VISIBLE_DEVICES=1

# Test Model MixBEATS
python -u test.py --config-file ./configs/Electricity/Electricity_96_24.json --model MixBEATS --model_save_path ./checkpoints/MixBEATS/Electricity_96_24 --result_path ./results/MixBEATS/Electricity_96_24
python -u test.py --config-file ./configs/ETTh1/ETTh1_96_24.json --model MixBEATS --model_save_path ./checkpoints/MixBEATS/ETTh1_96_24 --result_path ./results/MixBEATS/ETTh1_96_24
python -u test.py --config-file ./configs/ETTh2/ETTh2_96_24.json --model MixBEATS --model_save_path ./checkpoints/MixBEATS/ETTh2_96_24 --result_path ./results/MixBEATS/ETTh2_96_24
python -u test.py --config-file ./configs/ETTm1/ETTm1_96_24.json --model MixBEATS --model_save_path ./checkpoints/MixBEATS/ETTm1_96_24 --result_path ./results/MixBEATS/ETTm1_96_24
python -u test.py --config-file ./configs/ETTm2/ETTm2_96_24.json --model MixBEATS --model_save_path ./checkpoints/MixBEATS/ETTm2_96_24 --result_path ./results/MixBEATS/ETTm2_96_24
python -u test.py --config-file ./configs/Traffic/Traffic_96_24.json --model MixBEATS --model_save_path ./checkpoints/MixBEATS/Traffic_96_24 --result_path ./results/MixBEATS/Traffic_96_24
python -u test.py --config-file ./configs/Weather/Weather_96_24.json --model MixBEATS --model_save_path ./checkpoints/MixBEATS/Weather_96_24 --result_path ./results/MixBEATS/Weather_96_24
python -u test.py --config-file ./configs/Illness/Illness_36_24.json --model MixBEATS --model_save_path ./checkpoints/MixBEATS/Illness_36_24 --result_path ./results/MixBEATS/Illness_36_24
python -u test.py --config-file ./configs/Exchange/Exchange_96_24.json --model MixBEATS --model_save_path ./checkpoints/MixBEATS/Exchange_96_24 --result_path ./results/MixBEATS/Exchange_96_24
