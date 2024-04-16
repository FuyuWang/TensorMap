# TensorMap

TensorMap a deep reinforcement learning-based DNN mapper, 
to search the optimized mapping strategies for spatial accelerators.
This repository contains the source code for TensorMap.

### Setup ###
* Download the TensorMap source code
* Create virtual environment through anaconda
```
conda create --name TensorMapEnv python=3.8
conda activate TensorMapEnv
```
* Install packages
   
```
pip install -r requirements.txt
```

* Install [MAESTRO](https://github.com/maestro-project/maestro.git)
```
python build.py
```

### Run TensorMap ###

* Run RL-based mapping search of TensorMap on TPU
```
./run_tpu.sh
```

* Run RL of mapping search TensorMap on Eyeriss
```
./run_eyeriss.sh
```

* Run GA-based improvement of TensorMap
``