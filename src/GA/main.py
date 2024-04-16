import copy
import argparse
from datetime import datetime
import glob
import pandas as pd
import pickle
import os, sys
import torch

from env import MaestroEnvironment

script_dir = os.path.dirname(__file__)
module_path = os.path.abspath(os.path.join(script_dir, '../../'))
if module_path not in sys.path:
    sys.path.insert(0,module_path)
fitness_list = None
fitness = None
stage_idx = 0
prev_stage_value = []

def save_chkpt(layerid_to_dim,dim_info,dim_set,first_stage_value=None):
    chkpt = {
        "layerid_to_dim": layerid_to_dim,
        "dim_info": dim_info,
        "dim_set": dim_set,
        "first_stage_value":first_stage_value
    }
    with open(chkpt_file, "wb") as fd:
        pickle.dump(chkpt, fd)


def train_model(model_defs, stages=2, layer_id=None):
    layerid_to_dim = {}
    dim_infos = {}
    fitness_list = [opt.fitness, 'latency', 'energy']
    global fitness, prev_stage_value, stage_idx
    dim_set = set((tuple(m) for m in model_defs))
    dimension = model_defs[0]

    env = MaestroEnvironment(dimension=dimension, num_pe=opt.num_pe, fitness=fitness, par_RS=opt.parRS,
                      l1_size=opt.l1_size,
                      l2_size=opt.l2_size, NocBW = opt.NocBW, slevel_min=opt.slevel_min, slevel_max=opt.slevel_max,
                      fixedCluster=opt.fixedCluster, log_level=opt.log_level)

    for i, dim in enumerate(model_defs):
        layerid_to_dim[i] = dim

    for s in range(stages):
        dim_stage = {}
        for dimension in dim_set:
            chkpt_dir = "{}_SL-{}-{}_F-{}_PE-{}_L1-{}_L2-{}_EPOCH-{}/layer-{}".format(
                opt.model, opt.slevel_min, opt.slevel_max, opt.fitness, opt.num_pe,
                opt.l1_size, opt.l2_size, 10, layer_id
            )
            chkpt = torch.load(os.path.join(outdir, chkpt_dir, 'agent_chkpt.plt'))
            if s == 0:
                dim_stage[dimension] = {"best_reward": [chkpt['best_reward']],
                                        "best_sol": chkpt['best_sol']}
            else:
                dim_stage[dimension] = {"best_reward": [chkpt['best_reward'], float("-Inf")],
                                        "best_sol": chkpt['best_sol']}
        dim_infos["Stage{}".format(s+1)] = copy.deepcopy(dim_stage)
    for s in range(1, stages):
        stage_idx = s
        dim_list = list(dim_set)
        fitness = fitness_list
        chkpt_list = []
        for i, dimension in enumerate(dim_list):
            env.reset_dimension(dimension=dimension, fitness=fitness)
            chkpt_list.append(env.run(dimension, stage_idx=stage_idx, num_population=opt.num_pop,prev_stage_value=prev_stage_value,
                   num_generations=opt.epochs,best_sol_1st=dim_infos["Stage{}".format(s)][dimension]["best_sol"] if s!=0 else None))

        for i, chkpt in enumerate(chkpt_list):
            best_reward = chkpt["best_reward"]
            cur_best_reward =  dim_infos["Stage{}".format(s+1)][dim_list[i]]["best_reward"]
            if cur_best_reward[s] <= best_reward[s]:
                dim_infos["Stage{}".format(s+1)][dim_list[i]] = chkpt
        save_chkpt(layerid_to_dim, dim_infos, dim_set)
        if s+1==stages:
            return
        cur_stage_value = 0
        for dim in dim_set:
            cur_best_reward =  dim_infos["Stage{}".format(s+1)][dim]["best_reward"]
            cur_stage_value = min(cur_stage_value, cur_best_reward[stage_idx])
        prev_stage_value.append(cur_stage_value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fitness', type=str, default="latency", help='first stage fitness')
    parser.add_argument('--stages', type=int, default=2,help='number of stages', choices=[1,2])
    parser.add_argument('--num_pop', type=int, default=100,help='number of populations')
    parser.add_argument('--parRS', default=False, action='store_true', help='Parallize across R S dimension')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--outdir', type=str, default="outdir", help='output directiory')
    parser.add_argument('--num_pe', type=int, default=168, help='number of PEs')
    parser.add_argument('--l1_size', type=int, default=512, help='L1 size')
    parser.add_argument('--l2_size', type=int, default=108000, help='L2 size')
    parser.add_argument('--NocBW', type=int, default=8192000, help='NoC BW')
    parser.add_argument('--hwconfig', type=str, default=None, help='HW configuration file')
    parser.add_argument('--accelerator', type=str, default="TPU_bak", help='Accelerator to run')
    parser.add_argument('--model', type=str, default="vgg16", help='Model to run')
    parser.add_argument('--num_layer', type=int, default=0, help='number of layers to optimize')
    parser.add_argument('--slevel_min', type=int, default=2, help='parallelization level min')
    parser.add_argument('--slevel_max', type=int, default=2, help='parallelization level max')
    parser.add_argument('--fixedCluster', type=int, default=0, help='Rigid cluster size')
    parser.add_argument('--log_level', type=int, default=1, help='Detail: 2, runtimeinfo: 1')
    opt = parser.parse_args()
    m_file_path = "../../data/model/"
    m_file = os.path.join(m_file_path, opt.model + ".csv")
    df = pd.read_csv(m_file)
    model_defs = df.to_numpy()
    num_layers, dim_size = model_defs.shape

    for i in range(num_layers):
        dimension = model_defs[i:i+1]
        outdir = opt.outdir
        outdir = os.path.join("../../../", outdir)
        if opt.fixedCluster>0:
            exp_name = "GA_{}_{}_SL-{}-{}_F-{}_PE-{}_L1-{}_L2-{}_GEN-{}_POP-{}/layer-{}".format(opt.model, opt.accelerator, opt.slevel_min, opt.slevel_max,opt.fitness, opt.num_pe, opt.l1_size, opt.l2_size, opt.epochs, opt.num_pop, i)
        else:
            exp_name = "GA_{}_{}_SL-{}-{}_FixCl-{}_F2-{}_PE-{}_L1-{}_L2-{}_GEN-{}_POP-{}/layer-{}".format(opt.model, opt.accelerator, opt.slevel_min, opt.slevel_max,opt.fixedCluster, opt.fitness, opt.num_pe, opt.l1_size, opt.l2_size, opt.epochs, opt.num_pop, i)
        outdir_exp = os.path.join(outdir, exp_name)
        os.makedirs(outdir, exist_ok=True)
        os.makedirs(outdir_exp, exist_ok=True)
        chkpt_file_t = "{}".format("result")
        chkpt_file = os.path.join(outdir_exp, chkpt_file_t + "_c.plt")

        try:
            train_model(dimension, stages=opt.stages, layer_id=i)
        finally:
            for f in glob.glob("*.m"):
                os.remove(f)
            for f in glob.glob("*.csv"):
                os.remove(f)
