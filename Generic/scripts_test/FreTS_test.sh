export CUDA_VISIBLE_DEVICES=1

# Test Model FreTS
python -u test.py --config-file ./configs/Electricity/Electricity_96_24.json --model FreTS --model_save_path ./checkpoints/FreTS/Electricity_96_24 --result_path ./results/FreTS/Electricity_96_24
python -u test.py --config-file ./configs/ETTh1/ETTh1_96_24.json --model FreTS --model_save_path ./checkpoints/FreTS/ETTh1_96_24 --result_path ./results/FreTS/ETTh1_96_24
python -u test.py --config-file ./configs/ETTh2/ETTh2_96_24.json --model FreTS --model_save_path ./checkpoints/FreTS/ETTh2_96_24 --result_path ./results/FreTS/ETTh2_96_24
python -u test.py --config-file ./configs/ETTm1/ETTm1_96_24.json --model FreTS --model_save_path ./checkpoints/FreTS/ETTm1_96_24 --result_path ./results/FreTS/ETTm1_96_24
python -u test.py --config-file ./configs/ETTm2/ETTm2_96_24.json --model FreTS --model_save_path ./checkpoints/FreTS/ETTm2_96_24 --result_path ./results/FreTS/ETTm2_96_24
python -u test.py --config-file ./configs/Traffic/Traffic_96_24.json --model FreTS --model_save_path ./checkpoints/FreTS/Traffic_96_24 --result_path ./results/FreTS/Traffic_96_24
python -u test.py --config-file ./configs/Weather/Weather_96_24.json --model FreTS --model_save_path ./checkpoints/FreTS/Weather_96_24 --result_path ./results/FreTS/Weather_96_24
python -u test.py --config-file ./configs/Illness/Illness_36_24.json --model FreTS --model_save_path ./checkpoints/FreTS/Illness_36_24 --result_path ./results/FreTS/Illness_36_24
python -u test.py --config-file ./configs/Exchange/Exchange_96_24.json --model FreTS --model_save_path ./checkpoints/FreTS/Exchange_96_24 --result_path ./results/FreTS/Exchange_96_24
