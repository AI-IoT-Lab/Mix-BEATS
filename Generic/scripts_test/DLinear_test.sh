export CUDA_VISIBLE_DEVICES=0

# Test Model DLinear
python -u test.py --config-file ./configs/Electricity/Electricity_96_24.json --model DLinear --model_save_path ./checkpoints/DLinear/Electricity_96_24 --result_path ./results/DLinear/Electricity_96_24
python -u test.py --config-file ./configs/ETTh1/ETTh1_96_24.json --model DLinear --model_save_path ./checkpoints/DLinear/ETTh1_96_24 --result_path ./results/DLinear/ETTh1_96_24
python -u test.py --config-file ./configs/ETTh2/ETTh2_96_24.json --model DLinear --model_save_path ./checkpoints/DLinear/ETTh2_96_24 --result_path ./results/DLinear/ETTh2_96_24
python -u test.py --config-file ./configs/ETTm1/ETTm1_96_24.json --model DLinear --model_save_path ./checkpoints/DLinear/ETTm1_96_24 --result_path ./results/DLinear/ETTm1_96_24
python -u test.py --config-file ./configs/ETTm2/ETTm2_96_24.json --model DLinear --model_save_path ./checkpoints/DLinear/ETTm2_96_24 --result_path ./results/DLinear/ETTm2_96_24
python -u test.py --config-file ./configs/Traffic/Traffic_96_24.json --model DLinear --model_save_path ./checkpoints/DLinear/Traffic_96_24 --result_path ./results/DLinear/Traffic_96_24
python -u test.py --config-file ./configs/Weather/Weather_96_24.json --model DLinear --model_save_path ./checkpoints/DLinear/Weather_96_24 --result_path ./results/DLinear/Weather_96_24
python -u test.py --config-file ./configs/Illness/Illness_36_24.json --model DLinear --model_save_path ./checkpoints/DLinear/Illness_36_24 --result_path ./results/DLinear/Illness_36_24
python -u test.py --config-file ./configs/Exchange/Exchange_96_24.json --model DLinear --model_save_path ./checkpoints/DLinear/Exchange_96_24 --result_path ./results/DLinear/Exchange_96_24
