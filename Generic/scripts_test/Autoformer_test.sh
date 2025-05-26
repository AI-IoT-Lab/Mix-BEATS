export CUDA_VISIBLE_DEVICES=1


# Test Model Autoformer
python -u test.py --config-file ./configs/Electricity/Electricity_96_24.json --model Autoformer --model_save_path ./checkpoints/Autoformer/Electricity_96_24 --result_path ./results/Autoformer/Electricity_96_24
python -u test.py --config-file ./configs/ETTh1/ETTh1_96_24.json --model Autoformer --model_save_path ./checkpoints/Autoformer/ETTh1_96_24 --result_path ./results/Autoformer/ETTh1_96_24
python -u test.py --config-file ./configs/ETTh2/ETTh2_96_24.json --model Autoformer --model_save_path ./checkpoints/Autoformer/ETTh2_96_24 --result_path ./results/Autoformer/ETTh2_96_24
python -u test.py --config-file ./configs/ETTm1/ETTm1_96_24.json --model Autoformer --model_save_path ./checkpoints/Autoformer/ETTm1_96_24 --result_path ./results/Autoformer/ETTm1_96_24
python -u test.py --config-file ./configs/ETTm2/ETTm2_96_24.json --model Autoformer --model_save_path ./checkpoints/Autoformer/ETTm2_96_24 --result_path ./results/Autoformer/ETTm2_96_24
python -u test.py --config-file ./configs/Traffic/Traffic_96_24.json --model Autoformer --model_save_path ./checkpoints/Autoformer/Traffic_96_24 --result_path ./results/Autoformer/Traffic_96_24
python -u test.py --config-file ./configs/Weather/Weather_96_24.json --model Autoformer --model_save_path ./checkpoints/Autoformer/Weather_96_24 --result_path ./results/Autoformer/Weather_96_24
python -u test.py --config-file ./configs/Illness/Illness_36_24.json --model Autoformer --model_save_path ./checkpoints/Autoformer/Illness_36_24 --result_path ./results/Autoformer/Illness_36_24
python -u test.py --config-file ./configs/Exchange/Exchange_96_24.json --model Autoformer --model_save_path ./checkpoints/Autoformer/Exchange_96_24 --result_path ./results/Autoformer/Exchange_96_24
