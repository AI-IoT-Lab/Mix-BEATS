export CUDA_VISIBLE_DEVICES=1

# Test Model NBEATS
python -u test.py --config-file ./configs/Electricity/Electricity_96_24.json --model NBEATS --model_save_path ./checkpoints/NBEATS/Electricity_96_24 --result_path ./results/NBEATS/Electricity_96_24
python -u test.py --config-file ./configs/ETTh1/ETTh1_96_24.json --model NBEATS --model_save_path ./checkpoints/NBEATS/ETTh1_96_24 --result_path ./results/NBEATS/ETTh1_96_24
python -u test.py --config-file ./configs/ETTh2/ETTh2_96_24.json --model NBEATS --model_save_path ./checkpoints/NBEATS/ETTh2_96_24 --result_path ./results/NBEATS/ETTh2_96_24
python -u test.py --config-file ./configs/ETTm1/ETTm1_96_24.json --model NBEATS --model_save_path ./checkpoints/NBEATS/ETTm1_96_24 --result_path ./results/NBEATS/ETTm1_96_24
python -u test.py --config-file ./configs/ETTm2/ETTm2_96_24.json --model NBEATS --model_save_path ./checkpoints/NBEATS/ETTm2_96_24 --result_path ./results/NBEATS/ETTm2_96_24
python -u test.py --config-file ./configs/Traffic/Traffic_96_24.json --model NBEATS --model_save_path ./checkpoints/NBEATS/Traffic_96_24 --result_path ./results/NBEATS/Traffic_96_24
python -u test.py --config-file ./configs/Weather/Weather_96_24.json --model NBEATS --model_save_path ./checkpoints/NBEATS/Weather_96_24 --result_path ./results/NBEATS/Weather_96_24
python -u test.py --config-file ./configs/Illness/Illness_36_24.json --model NBEATS --model_save_path ./checkpoints/NBEATS/Illness_36_24 --result_path ./results/NBEATS/Illness_36_24
python -u test.py --config-file ./configs/Exchange/Exchange_96_24.json --model NBEATS --model_save_path ./checkpoints/NBEATS/Exchange_96_24 --result_path ./results/NBEATS/Exchange_96_24
