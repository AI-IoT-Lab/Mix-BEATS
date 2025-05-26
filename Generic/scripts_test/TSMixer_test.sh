export CUDA_VISIBLE_DEVICES=1

# Test Model TSMixer
python -u test.py --config-file ./configs/Electricity/Electricity_96_24.json --model TSMixer --model_save_path ./checkpoints/TSMixer/Electricity_96_24 --result_path ./results/TSMixer/Electricity_96_24
python -u test.py --config-file ./configs/ETTh1/ETTh1_96_24.json --model TSMixer --model_save_path ./checkpoints/TSMixer/ETTh1_96_24 --result_path ./results/TSMixer/ETTh1_96_24
python -u test.py --config-file ./configs/ETTh2/ETTh2_96_24.json --model TSMixer --model_save_path ./checkpoints/TSMixer/ETTh2_96_24 --result_path ./results/TSMixer/ETTh2_96_24
python -u test.py --config-file ./configs/ETTm1/ETTm1_96_24.json --model TSMixer --model_save_path ./checkpoints/TSMixer/ETTm1_96_24 --result_path ./results/TSMixer/ETTm1_96_24
python -u test.py --config-file ./configs/ETTm2/ETTm2_96_24.json --model TSMixer --model_save_path ./checkpoints/TSMixer/ETTm2_96_24 --result_path ./results/TSMixer/ETTm2_96_24
python -u test.py --config-file ./configs/Traffic/Traffic_96_24.json --model TSMixer --model_save_path ./checkpoints/TSMixer/Traffic_96_24 --result_path ./results/TSMixer/Traffic_96_24
python -u test.py --config-file ./configs/Weather/Weather_96_24.json --model TSMixer --model_save_path ./checkpoints/TSMixer/Weather_96_24 --result_path ./results/TSMixer/Weather_96_24
python -u test.py --config-file ./configs/Illness/Illness_36_24.json --model TSMixer --model_save_path ./checkpoints/TSMixer/Illness_36_24 --result_path ./results/TSMixer/Illness_36_24
python -u test.py --config-file ./configs/Exchange/Exchange_96_24.json --model TSMixer --model_save_path ./checkpoints/TSMixer/Exchange_96_24 --result_path ./results/TSMixer/Exchange_96_24
