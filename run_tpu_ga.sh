cd ./src/GA
python main.py --fitness EDP --stages 2 --accelerator TPU --num_pe 65536 --l1_size 65536 --l2_size 25165824 --slevel_min 2 --slevel_max 3 --epochs 20 --num_pop 10 --model ResNet50
cd ../..






