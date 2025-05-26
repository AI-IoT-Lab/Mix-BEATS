export CUDA_VISIBLE_DEVICES=0

# Test Model PatchTST
python -u test.py --config-file ./configs/Electricity/Electricity_96_24.json --model PatchTST --model_save_path ./checkpoints/PatchTST/Electricity_96_24 --result_path ./results/PatchTST/Electricity_96_24
python -u test.py --config-file ./configs/ETTh1/ETTh1_96_24.json --model PatchTST --model_save_path ./checkpoints/PatchTST/ETTh1_96_24 --result_path ./results/PatchTST/ETTh1_96_24
python -u test.py --config-file ./configs/ETTh2/ETTh2_96_24.json --model PatchTST --model_save_path ./checkpoints/PatchTST/ETTh2_96_24 --result_path ./results/PatchTST/ETTh2_96_24
python -u test.py --config-file ./configs/ETTm1/ETTm1_96_24.json --model PatchTST --model_save_path ./checkpoints/PatchTST/ETTm1_96_24 --result_path ./results/PatchTST/ETTm1_96_24
python -u test.py --config-file ./configs/ETTm2/ETTm2_96_24.json --model PatchTST --model_save_path ./checkpoints/PatchTST/ETTm2_96_24 --result_path ./results/PatchTST/ETTm2_96_24
python -u test.py --config-file ./configs/Traffic/Traffic_96_24.json --model PatchTST --model_save_path ./checkpoints/PatchTST/Traffic_96_24 --result_path ./results/PatchTST/Traffic_96_24
python -u test.py --config-file ./configs/Weather/Weather_96_24.json --model PatchTST --model_save_path ./checkpoints/PatchTST/Weather_96_24 --result_path ./results/PatchTST/Weather_96_24
python -u test.py --config-file ./configs/Illness/Illness_36_24.json --model PatchTST --model_save_path ./checkpoints/PatchTST/Illness_36_24 --result_path ./results/PatchTST/Illness_36_24
python -u test.py --config-file ./configs/Exchange/Exchange_96_24.json --model PatchTST --model_save_path ./checkpoints/PatchTST/Exchange_96_24 --result_path ./results/PatchTST/Exchange_96_24
