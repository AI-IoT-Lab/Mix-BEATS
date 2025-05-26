export CUDA_VISIBLE_DEVICES=1

# Test Model iTransformer
python -u test.py --config-file ./configs/Electricity/Electricity_96_24.json --model iTransformer --model_save_path ./checkpoints/iTransformer/Electricity_96_24 --result_path ./results/iTransformer/Electricity_96_24
python -u test.py --config-file ./configs/ETTh1/ETTh1_96_24.json --model iTransformer --model_save_path ./checkpoints/iTransformer/ETTh1_96_24 --result_path ./results/iTransformer/ETTh1_96_24
python -u test.py --config-file ./configs/ETTh2/ETTh2_96_24.json --model iTransformer --model_save_path ./checkpoints/iTransformer/ETTh2_96_24 --result_path ./results/iTransformer/ETTh2_96_24
python -u test.py --config-file ./configs/ETTm1/ETTm1_96_24.json --model iTransformer --model_save_path ./checkpoints/iTransformer/ETTm1_96_24 --result_path ./results/iTransformer/ETTm1_96_24
python -u test.py --config-file ./configs/ETTm2/ETTm2_96_24.json --model iTransformer --model_save_path ./checkpoints/iTransformer/ETTm2_96_24 --result_path ./results/iTransformer/ETTm2_96_24
python -u test.py --config-file ./configs/Traffic/Traffic_96_24.json --model iTransformer --model_save_path ./checkpoints/iTransformer/Traffic_96_24 --result_path ./results/iTransformer/Traffic_96_24
python -u test.py --config-file ./configs/Weather/Weather_96_24.json --model iTransformer --model_save_path ./checkpoints/iTransformer/Weather_96_24 --result_path ./results/iTransformer/Weather_96_24
python -u test.py --config-file ./configs/Illness/Illness_36_24.json --model iTransformer --model_save_path ./checkpoints/iTransformer/Illness_36_24 --result_path ./results/iTransformer/Illness_36_24
python -u test.py --config-file ./configs/Exchange/Exchange_96_24.json --model iTransformer --model_save_path ./checkpoints/iTransformer/Exchange_96_24 --result_path ./results/iTransformer/Exchange_96_24
