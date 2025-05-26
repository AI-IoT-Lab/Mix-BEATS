export CUDA_VISIBLE_DEVICES=1

# Test Model LightTS
python -u test.py --config-file ./configs/Electricity/Electricity_96_24.json --model LightTS --model_save_path ./checkpoints/LightTS/Electricity_96_24 --result_path ./results/LightTS/Electricity_96_24
python -u test.py --config-file ./configs/ETTh1/ETTh1_96_24.json --model LightTS --model_save_path ./checkpoints/LightTS/ETTh1_96_24 --result_path ./results/LightTS/ETTh1_96_24
python -u test.py --config-file ./configs/ETTh2/ETTh2_96_24.json --model LightTS --model_save_path ./checkpoints/LightTS/ETTh2_96_24 --result_path ./results/LightTS/ETTh2_96_24
python -u test.py --config-file ./configs/ETTm1/ETTm1_96_24.json --model LightTS --model_save_path ./checkpoints/LightTS/ETTm1_96_24 --result_path ./results/LightTS/ETTm1_96_24
python -u test.py --config-file ./configs/ETTm2/ETTm2_96_24.json --model LightTS --model_save_path ./checkpoints/LightTS/ETTm2_96_24 --result_path ./results/LightTS/ETTm2_96_24
python -u test.py --config-file ./configs/Traffic/Traffic_96_24.json --model LightTS --model_save_path ./checkpoints/LightTS/Traffic_96_24 --result_path ./results/LightTS/Traffic_96_24
python -u test.py --config-file ./configs/Weather/Weather_96_24.json --model LightTS --model_save_path ./checkpoints/LightTS/Weather_96_24 --result_path ./results/LightTS/Weather_96_24
python -u test.py --config-file ./configs/Illness/Illness_36_24.json --model LightTS --model_save_path ./checkpoints/LightTS/Illness_36_24 --result_path ./results/LightTS/Illness_36_24
python -u test.py --config-file ./configs/Exchange/Exchange_96_24.json --model LightTS --model_save_path ./checkpoints/LightTS/Exchange_96_24 --result_path ./results/LightTS/Exchange_96_24

