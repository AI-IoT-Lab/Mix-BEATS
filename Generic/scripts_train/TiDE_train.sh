export CUDA_VISIBLE_DEVICES=1

# Train Electricity
python -u train.py --config-file ./configs/Electricity/Electricity_96_24.json --model TiDE  --model_save_path ./checkpoints/TiDE/Electricity_96_24
# python -u train.py --config-file ./configs/Electricity/Electricity_96_96.json

# Train ETTh1
python -u train.py --config-file ./configs/ETTh1/ETTh1_96_24.json --model TiDE --model_save_path ./checkpoints/TiDE/ETTh1_96_24
# python -u train.py --config-file ./configs/ETTh1/ETTh1_96_96.json

# Train ETTh2
python -u train.py --config-file ./configs/ETTh2/ETTh2_96_24.json --model TiDE --model_save_path ./checkpoints/TiDE/ETTh2_96_24
# python -u train.py --config-file ./configs/ETTh2/ETTh2_96_96.json


# Train ETTm1
python -u train.py --config-file ./configs/ETTm1/ETTm1_96_24.json --model TiDE --model_save_path ./checkpoints/TiDE/ETTm1_96_24
# python -u train.py --config-file ./configs/ETTm1/ETTm1_96_96.json

#Train ETTm2
python -u train.py --config-file ./configs/ETTm2/ETTm2_96_24.json --model TiDE --model_save_path ./checkpoints/TiDE/ETTm2_96_24
# python -u train.py --config-file ./configs/ETTm2/ETTm2_96_96.json

# Train Exchange
python -u train.py --config-file ./configs/Exchange/Exchange_96_24.json --model TiDE --model_save_path ./checkpoints/TiDE/Exchange_96_24
# python -u train.py --config-file ./configs/Exchange/Exchange_96_96.json

# Train Illness
python -u train.py --config-file ./configs/Illness/Illness_36_24.json --model TiDE --model_save_path ./checkpoints/TiDE/Illness_36_24
# python -u train.py --config-file ./configs/Illness/Illness_36_36.json

# Train Traffic
python -u train.py --config-file ./configs/Traffic/Traffic_96_24.json --model TiDE --model_save_path ./checkpoints/TiDE/Traffic_96_24
# python -u train.py --config-file ./configs/Traffic/Traffic_96_96.json

# Train Weather
python -u train.py --config-file ./configs/Weather/Weather_96_24.json --model TiDE --model_save_path ./checkpoints/TiDE/Weather_96_24
# python -u train.py --config-file ./configs/Weather/Weather_96_96.json
