export CUDA_VISIBLE_DEVICES=0

# Exp Electricity
python -u exp.py --config-file ./configs/Electricity/Electricity_96_24.json --model MixBEATS
