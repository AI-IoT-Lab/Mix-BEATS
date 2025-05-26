export CUDA_VISIBLE_DEVICES=1

# Test Model SegRNN
python -u test.py --config-file ./configs/Electricity/Electricity_96_24.json --model SegRNN --model_save_path ./checkpoints/SegRNN/Electricity_96_24 --result_path ./results/SegRNN/Electricity_96_24
python -u test.py --config-file ./configs/ETTh1/ETTh1_96_24.json --model SegRNN --model_save_path ./checkpoints/SegRNN/ETTh1_96_24 --result_path ./results/SegRNN/ETTh1_96_24
python -u test.py --config-file ./configs/ETTh2/ETTh2_96_24.json --model SegRNN --model_save_path ./checkpoints/SegRNN/ETTh2_96_24 --result_path ./results/SegRNN/ETTh2_96_24
python -u test.py --config-file ./configs/ETTm1/ETTm1_96_24.json --model SegRNN --model_save_path ./checkpoints/SegRNN/ETTm1_96_24 --result_path ./results/SegRNN/ETTm1_96_24
python -u test.py --config-file ./configs/ETTm2/ETTm2_96_24.json --model SegRNN --model_save_path ./checkpoints/SegRNN/ETTm2_96_24 --result_path ./results/SegRNN/ETTm2_96_24
python -u test.py --config-file ./configs/Traffic/Traffic_96_24.json --model SegRNN --model_save_path ./checkpoints/SegRNN/Traffic_96_24 --result_path ./results/SegRNN/Traffic_96_24
python -u test.py --config-file ./configs/Weather/Weather_96_24.json --model SegRNN --model_save_path ./checkpoints/SegRNN/Weather_96_24 --result_path ./results/SegRNN/Weather_96_24
python -u test.py --config-file ./configs/Illness/Illness_36_24.json --model SegRNN --model_save_path ./checkpoints/SegRNN/Illness_36_24 --result_path ./results/SegRNN/Illness_36_24
python -u test.py --config-file ./configs/Exchange/Exchange_96_24.json --model SegRNN --model_save_path ./checkpoints/SegRNN/Exchange_96_24 --result_path ./results/SegRNN/Exchange_96_24
