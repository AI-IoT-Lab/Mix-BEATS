export CUDA_VISIBLE_DEVICES=1

# Train Electricity
python -u train.py --config-file ./configs/Electricity/Electricity_96_24.json --model SegRNN  --model_save_path ./checkpoints/SegRNN/Electricity_96_24
# python -u train.py --config-file ./configs/Electricity/Electricity_96_96.json

# Train ETTh1
python -u train.py --config-file ./configs/ETTh1/ETTh1_96_24.json --model SegRNN --model_save_path ./checkpoints/SegRNN/ETTh1_96_24
# python -u train.py --config-file ./configs/ETTh1/ETTh1_96_96.json

# Train ETTh2
python -u train.py --config-file ./configs/ETTh2/ETTh2_96_24.json --model SegRNN --model_save_path ./checkpoints/SegRNN/ETTh2_96_24
# python -u train.py --config-file ./configs/ETTh2/ETTh2_96_96.json


# Train ETTm1
python -u train.py --config-file ./configs/ETTm1/ETTm1_96_24.json --model SegRNN --model_save_path ./checkpoints/SegRNN/ETTm1_96_24
# python -u train.py --config-file ./configs/ETTm1/ETTm1_96_96.json

#Train ETTm2
python -u train.py --config-file ./configs/ETTm2/ETTm2_96_24.json --model SegRNN --model_save_path ./checkpoints/SegRNN/ETTm2_96_24
# python -u train.py --config-file ./configs/ETTm2/ETTm2_96_96.json

# Train Exchange
python -u train.py --config-file ./configs/Exchange/Exchange_96_24.json --model SegRNN --model_save_path ./checkpoints/SegRNN/Exchange_96_24
# python -u train.py --config-file ./configs/Exchange/Exchange_96_96.json

# Train Illness
python -u train.py --config-file ./configs/Illness/Illness_36_24.json --model SegRNN --model_save_path ./checkpoints/SegRNN/Illness_36_24
# python -u train.py --config-file ./configs/Illness/Illness_36_36.json

# Train Traffic
python -u train.py --config-file ./configs/Traffic/Traffic_96_24.json --model SegRNN --model_save_path ./checkpoints/SegRNN/Traffic_96_24
# python -u train.py --config-file ./configs/Traffic/Traffic_96_96.json

# Train Weather
python -u train.py --config-file ./configs/Weather/Weather_96_24.json --model SegRNN --model_save_path ./checkpoints/SegRNN/Weather_96_24
# python -u train.py --config-file ./configs/Weather/Weather_96_96.json
