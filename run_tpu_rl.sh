cd ./src/RL
CUDA_VISIBLE_DEVICES=0 python main.py --fitness EDP --num_pe 65536 --l1_size 65536 --l2_size 25165824 --slevel_min 2 --slevel_max 3 --epochs 10 --model ResNet50
cd ../..