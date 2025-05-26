export CUDA_VISIBLE_DEVICES=1


# Test Model SCINet
python -u test.py --config-file ./configs/Electricity/Electricity_96_24.json --model SCINet --model_save_path ./checkpoints/SCINet/Electricity_96_24 --result_path ./results/SCINet/Electricity_96_24
python -u test.py --config-file ./configs/ETTh1/ETTh1_96_24.json --model SCINet --model_save_path ./checkpoints/SCINet/ETTh1_96_24 --result_path ./results/SCINet/ETTh1_96_24
python -u test.py --config-file ./configs/ETTh2/ETTh2_96_24.json --model SCINet --model_save_path ./checkpoints/SCINet/ETTh2_96_24 --result_path ./results/SCINet/ETTh2_96_24
python -u test.py --config-file ./configs/ETTm1/ETTm1_96_24.json --model SCINet --model_save_path ./checkpoints/SCINet/ETTm1_96_24 --result_path ./results/SCINet/ETTm1_96_24
python -u test.py --config-file ./configs/ETTm2/ETTm2_96_24.json --model SCINet --model_save_path ./checkpoints/SCINet/ETTm2_96_24 --result_path ./results/SCINet/ETTm2_96_24
python -u test.py --config-file ./configs/Traffic/Traffic_96_24.json --model SCINet --model_save_path ./checkpoints/SCINet/Traffic_96_24 --result_path ./results/SCINet/Traffic_96_24
python -u test.py --config-file ./configs/Weather/Weather_96_24.json --model SCINet --model_save_path ./checkpoints/SCINet/Weather_96_24 --result_path ./results/SCINet/Weather_96_24
python -u test.py --config-file ./configs/Illness/Illness_36_24.json --model SCINet --model_save_path ./checkpoints/SCINet/Illness_36_24 --result_path ./results/SCINet/Illness_36_24
python -u test.py --config-file ./configs/Exchange/Exchange_96_24.json --model SCINet --model_save_path ./checkpoints/SCINet/Exchange_96_24 --result_path ./results/SCINet/Exchange_96_24
