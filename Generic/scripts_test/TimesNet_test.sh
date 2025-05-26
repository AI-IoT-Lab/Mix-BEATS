export CUDA_VISIBLE_DEVICES=0

# Test Model TimesNet
python -u test.py --config-file ./configs/Electricity/Electricity_96_24.json --model TimesNet --model_save_path ./checkpoints/TimesNet/Electricity_96_24 --result_path ./results/TimesNet/Electricity_96_24
python -u test.py --config-file ./configs/ETTh1/ETTh1_96_24.json --model TimesNet --model_save_path ./checkpoints/TimesNet/ETTh1_96_24 --result_path ./results/TimesNet/ETTh1_96_24
python -u test.py --config-file ./configs/ETTh2/ETTh2_96_24.json --model TimesNet --model_save_path ./checkpoints/TimesNet/ETTh2_96_24 --result_path ./results/TimesNet/ETTh2_96_24
python -u test.py --config-file ./configs/ETTm1/ETTm1_96_24.json --model TimesNet --model_save_path ./checkpoints/TimesNet/ETTm1_96_24 --result_path ./results/TimesNet/ETTm1_96_24
python -u test.py --config-file ./configs/ETTm2/ETTm2_96_24.json --model TimesNet --model_save_path ./checkpoints/TimesNet/ETTm2_96_24 --result_path ./results/TimesNet/ETTm2_96_24
python -u test.py --config-file ./configs/Traffic/Traffic_96_24.json --model TimesNet --model_save_path ./checkpoints/TimesNet/Traffic_96_24 --result_path ./results/TimesNet/Traffic_96_24
python -u test.py --config-file ./configs/Weather/Weather_96_24.json --model TimesNet --model_save_path ./checkpoints/TimesNet/Weather_96_24 --result_path ./results/TimesNet/Weather_96_24
python -u test.py --config-file ./configs/Illness/Illness_36_24.json --model TimesNet --model_save_path ./checkpoints/TimesNet/Illness_36_24 --result_path ./results/TimesNet/Illness_36_24
python -u test.py --config-file ./configs/Exchange/Exchange_96_24.json --model TimesNet --model_save_path ./checkpoints/TimesNet/Exchange_96_24 --result_path ./results/TimesNet/Exchange_96_24

