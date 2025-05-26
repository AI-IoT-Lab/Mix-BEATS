export CUDA_VISIBLE_DEVICES=1

# Test Model Informer
python -u test.py --config-file ./configs/Electricity/Electricity_96_24.json --model Informer --model_save_path ./checkpoints/Informer/Electricity_96_24 --result_path ./results/Informer/Electricity_96_24
python -u test.py --config-file ./configs/ETTh1/ETTh1_96_24.json --model Informer --model_save_path ./checkpoints/Informer/ETTh1_96_24 --result_path ./results/Informer/ETTh1_96_24
python -u test.py --config-file ./configs/ETTh2/ETTh2_96_24.json --model Informer --model_save_path ./checkpoints/Informer/ETTh2_96_24 --result_path ./results/Informer/ETTh2_96_24
python -u test.py --config-file ./configs/ETTm1/ETTm1_96_24.json --model Informer --model_save_path ./checkpoints/Informer/ETTm1_96_24 --result_path ./results/Informer/ETTm1_96_24
python -u test.py --config-file ./configs/ETTm2/ETTm2_96_24.json --model Informer --model_save_path ./checkpoints/Informer/ETTm2_96_24 --result_path ./results/Informer/ETTm2_96_24
python -u test.py --config-file ./configs/Traffic/Traffic_96_24.json --model Informer --model_save_path ./checkpoints/Informer/Traffic_96_24 --result_path ./results/Informer/Traffic_96_24
python -u test.py --config-file ./configs/Weather/Weather_96_24.json --model Informer --model_save_path ./checkpoints/Informer/Weather_96_24 --result_path ./results/Informer/Weather_96_24
python -u test.py --config-file ./configs/Illness/Illness_36_24.json --model Informer --model_save_path ./checkpoints/Informer/Illness_36_24 --result_path ./results/Informer/Illness_36_24
python -u test.py --config-file ./configs/Exchange/Exchange_96_24.json --model Informer --model_save_path ./checkpoints/Informer/Exchange_96_24 --result_path ./results/Informer/Exchange_96_24
