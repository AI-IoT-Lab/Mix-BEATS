export CUDA_VISIBLE_DEVICES=1

# Test Model Transformer

python -u test.py --config-file ./configs/Electricity/Electricity_96_24.json --model Transformer --model_save_path ./checkpoints/Transformer/Electricity_96_24 --result_path ./results/Transformer/Electricity_96_24
python -u test.py --config-file ./configs/ETTh1/ETTh1_96_24.json --model Transformer --model_save_path ./checkpoints/Transformer/ETTh1_96_24 --result_path ./results/Transformer/ETTh1_96_24
python -u test.py --config-file ./configs/ETTh2/ETTh2_96_24.json --model Transformer --model_save_path ./checkpoints/Transformer/ETTh2_96_24 --result_path ./results/Transformer/ETTh2_96_24
python -u test.py --config-file ./configs/ETTm1/ETTm1_96_24.json --model Transformer --model_save_path ./checkpoints/Transformer/ETTm1_96_24 --result_path ./results/Transformer/ETTm1_96_24
python -u test.py --config-file ./configs/ETTm2/ETTm2_96_24.json --model Transformer --model_save_path ./checkpoints/Transformer/ETTm2_96_24 --result_path ./results/Transformer/ETTm2_96_24
python -u test.py --config-file ./configs/Traffic/Traffic_96_24.json --model Transformer --model_save_path ./checkpoints/Transformer/Traffic_96_24 --result_path ./results/Transformer/Traffic_96_24
python -u test.py --config-file ./configs/Weather/Weather_96_24.json --model Transformer --model_save_path ./checkpoints/Transformer/Weather_96_24 --result_path ./results/Transformer/Weather_96_24
python -u test.py --config-file ./configs/Illness/Illness_36_24.json --model Transformer --model_save_path ./checkpoints/Transformer/Illness_36_24 --result_path ./results/Transformer/Illness_36_24
python -u test.py --config-file ./configs/Exchange/Exchange_96_24.json --model Transformer --model_save_path ./checkpoints/Transformer/Exchange_96_24 --result_path ./results/Transformer/Exchange_96_24