cd ./src/RL
CUDA_VISIBLE_DEVICES=1 python main.py --fitness EDP --num_pe 256 --l1_size 4096 --l2_size 131072 --slevel_min 2 --slevel_max 2 --epochs 10 --model ResNet50
cd ../..
