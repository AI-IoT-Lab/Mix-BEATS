export CUDA_VISIBLE_DEVICES=1

# Test Model Reformer
python -u test.py --config-file ./configs/Electricity/Electricity_96_24.json --model Reformer --model_save_path ./checkpoints/Reformer/Electricity_96_24 --result_path ./results/Reformer/Electricity_96_24
python -u test.py --config-file ./configs/ETTh1/ETTh1_96_24.json --model Reformer --model_save_path ./checkpoints/Reformer/ETTh1_96_24 --result_path ./results/Reformer/ETTh1_96_24
python -u test.py --config-file ./configs/ETTh2/ETTh2_96_24.json --model Reformer --model_save_path ./checkpoints/Reformer/ETTh2_96_24 --result_path ./results/Reformer/ETTh2_96_24
python -u test.py --config-file ./configs/ETTm1/ETTm1_96_24.json --model Reformer --model_save_path ./checkpoints/Reformer/ETTm1_96_24 --result_path ./results/Reformer/ETTm1_96_24
python -u test.py --config-file ./configs/ETTm2/ETTm2_96_24.json --model Reformer --model_save_path ./checkpoints/Reformer/ETTm2_96_24 --result_path ./results/Reformer/ETTm2_96_24
python -u test.py --config-file ./configs/Traffic/Traffic_96_24.json --model Reformer --model_save_path ./checkpoints/Reformer/Traffic_96_24 --result_path ./results/Reformer/Traffic_96_24
python -u test.py --config-file ./configs/Weather/Weather_96_24.json --model Reformer --model_save_path ./checkpoints/Reformer/Weather_96_24 --result_path ./results/Reformer/Weather_96_24
python -u test.py --config-file ./configs/Illness/Illness_36_24.json --model Reformer --model_save_path ./checkpoints/Reformer/Illness_36_24 --result_path ./results/Reformer/Illness_36_24
python -u test.py --config-file ./configs/Exchange/Exchange_96_24.json --model Reformer --model_save_path ./checkpoints/Reformer/Exchange_96_24 --result_path ./results/Reformer/Exchange_96_24
