export CUDA_VISIBLE_DEVICES=1

# Test Model Pyraformer
python -u test.py --config-file ./configs/Electricity/Electricity_96_24.json --model Pyraformer --model_save_path ./checkpoints/Pyraformer/Electricity_96_24 --result_path ./results/Pyraformer/Electricity_96_24
python -u test.py --config-file ./configs/ETTh1/ETTh1_96_24.json --model Pyraformer --model_save_path ./checkpoints/Pyraformer/ETTh1_96_24 --result_path ./results/Pyraformer/ETTh1_96_24
python -u test.py --config-file ./configs/ETTh2/ETTh2_96_24.json --model Pyraformer --model_save_path ./checkpoints/Pyraformer/ETTh2_96_24 --result_path ./results/Pyraformer/ETTh2_96_24
python -u test.py --config-file ./configs/ETTm1/ETTm1_96_24.json --model Pyraformer --model_save_path ./checkpoints/Pyraformer/ETTm1_96_24 --result_path ./results/Pyraformer/ETTm1_96_24
python -u test.py --config-file ./configs/ETTm2/ETTm2_96_24.json --model Pyraformer --model_save_path ./checkpoints/Pyraformer/ETTm2_96_24 --result_path ./results/Pyraformer/ETTm2_96_24
python -u test.py --config-file ./configs/Traffic/Traffic_96_24.json --model Pyraformer --model_save_path ./checkpoints/Pyraformer/Traffic_96_24 --result_path ./results/Pyraformer/Traffic_96_24
python -u test.py --config-file ./configs/Weather/Weather_96_24.json --model Pyraformer --model_save_path ./checkpoints/Pyraformer/Weather_96_24 --result_path ./results/Pyraformer/Weather_96_24
python -u test.py --config-file ./configs/Illness/Illness_36_24.json --model Pyraformer --model_save_path ./checkpoints/Pyraformer/Illness_36_24 --result_path ./results/Pyraformer/Illness_36_24
python -u test.py --config-file ./configs/Exchange/Exchange_96_24.json --model Pyraformer --model_save_path ./checkpoints/Pyraformer/Exchange_96_24 --result_path ./results/Pyraformer/Exchange_96_24
