export CUDA_VISIBLE_DEVICES=1


# Test Model TiDE
python -u test.py --config-file ./configs/Electricity/Electricity_96_24.json --model TiDE --model_save_path ./checkpoints/TiDE/Electricity_96_24 --result_path ./results/TiDE/Electricity_96_24
python -u test.py --config-file ./configs/ETTh1/ETTh1_96_24.json --model TiDE --model_save_path ./checkpoints/TiDE/ETTh1_96_24 --result_path ./results/TiDE/ETTh1_96_24
python -u test.py --config-file ./configs/ETTh2/ETTh2_96_24.json --model TiDE --model_save_path ./checkpoints/TiDE/ETTh2_96_24 --result_path ./results/TiDE/ETTh2_96_24
python -u test.py --config-file ./configs/ETTm1/ETTm1_96_24.json --model TiDE --model_save_path ./checkpoints/TiDE/ETTm1_96_24 --result_path ./results/TiDE/ETTm1_96_24
python -u test.py --config-file ./configs/ETTm2/ETTm2_96_24.json --model TiDE --model_save_path ./checkpoints/TiDE/ETTm2_96_24 --result_path ./results/TiDE/ETTm2_96_24
python -u test.py --config-file ./configs/Traffic/Traffic_96_24.json --model TiDE --model_save_path ./checkpoints/TiDE/Traffic_96_24 --result_path ./results/TiDE/Traffic_96_24
python -u test.py --config-file ./configs/Weather/Weather_96_24.json --model TiDE --model_save_path ./checkpoints/TiDE/Weather_96_24 --result_path ./results/TiDE/Weather_96_24
python -u test.py --config-file ./configs/Illness/Illness_36_24.json --model TiDE --model_save_path ./checkpoints/TiDE/Illness_36_24 --result_path ./results/TiDE/Illness_36_24
python -u test.py --config-file ./configs/Exchange/Exchange_96_24.json --model TiDE --model_save_path ./checkpoints/TiDE/Exchange_96_24 --result_path ./results/TiDE/Exchange_96_24
