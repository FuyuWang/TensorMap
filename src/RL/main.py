import copy
import argparse
import random
import pandas as pd
import glob
import os, sys
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim

from env import MaestroEnvironment
from actor import Actor
script_dir = os.path.dirname(__file__)
module_path = os.path.abspath(os.path.join(script_dir, '../../'))
if module_path not in sys.path:
    sys.path.insert(0,module_path)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LR_ACTOR = 1e-3  # learning rate of the actor
GAMMA = 0.9  # discount factor
CLIPPING_LSTM = 10
CLIPPING_MODEL = 100
EPISIOLON = 2**(-12)


def compute_policy_loss(rewards, log_probs, log_prob_masks):
    dis_rewards = []
    batch_size = log_probs.size(1)
    batch_masks = log_probs.new_ones(batch_size)
    # success_idx = []
    # fail_idx = []
    # for i in range(batch_size):
    #     if rewards[-1, i] > 0:
    #         success_idx.append(i)
    #     else:
    #         fail_idx.append(i)
    # if len(fail_idx) > 3 * len(success_idx):
    #     fail_idx = random.sample(fail_idx, 3 * len(success_idx))
    # print(len(success_idx), len(fail_idx), rewards[-1, :])
    # batch_masks = log_probs.new_zeros(batch_size)
    # batch_masks[success_idx] = 1.
    # batch_masks[fail_idx] = 1.

    # rewards = rewards[7:]
    # log_probs = log_probs[:-7]
    # log_prob_masks = log_prob_masks[:-7]

    R = np.zeros(batch_size)
    for r in rewards[::-1]:
        R = r + GAMMA * R
        dis_rewards.insert(0, R)
    dis_rewards = torch.from_numpy(np.array(dis_rewards)).to(log_probs.device)
    # print(dis_rewards.size(), log_prob_masks[:,7,:], log_prob_masks[:,6,:])
    policy_loss = dis_rewards * (-1 * log_probs * log_prob_masks).sum(dim=-1)
    policy_loss = policy_loss.sum(dim=0) * batch_masks
    policy_loss = policy_loss.sum() / batch_masks.sum()

    return policy_loss

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #  torch.backends.cudnn.deterministic = True


def train(dimension):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    LR_ACTOR = 1e-3  # learning rate of the actor
    CLIPPING_MODEL = 100

    agent_chkpt = {}
    agent_chkpt['dimension'] = dimension
    agent_chkpt['best_reward_record'] = []
    agent_chkpt['best_latency_record'] = []
    agent_chkpt['best_energy_record'] = []
    agent_chkpt['best_sols'] = []
    agent_chkpt['best_reward'] = float("-Inf")
    agent_chkpt['best_latency'] = float("-Inf")
    agent_chkpt['best_energy'] = float("-Inf")
    agent_chkpt['best_sol'] = float("-Inf")
    agent_chkpt['best_state'] = None

    num_episodes = 10
    NocBW = 2 * 4 * 8
    env = MaestroEnvironment(dimension=dimension, fitness=opt.fitness, par_RS=opt.parRS,
                             num_pe=opt.num_pe, l1_size=opt.l1_size, l2_size=opt.l2_size,
                             NocBW=NocBW, slevel_min=opt.slevel_min, slevel_max=opt.slevel_max,
                             log_level=opt.log_level, batch_size=num_episodes)

    actor = Actor(dimension, slevel_max=opt.slevel_max, slevel_min=opt.slevel_min,
                  par_RS=opt.parRS, batch_size=num_episodes).to(device)

    actor_optimizer = optim.Adam(actor.parameters(), lr=LR_ACTOR, betas=(0.9, 0.999))
    best_reward = float('-inf')
    thres_epoch = 10
    for ep in range(opt.epochs):
        policy_loss = 0.

        if ep > 0 and ep % thres_epoch == 0:
            for param_group in actor_optimizer.param_groups:
                for param in param_group['params']:
                    if param.requires_grad:
                        param_group['lr'] = param_group['lr'] * 0.8
                        break
                print(param_group['lr'])
        print(datetime.now().time())
        rewards = []
        log_probs = []
        log_prob_masks = []
        state_info = env.epoch_reset(dimension, opt.fitness)
        actor.reset()
        for t in range(1000):
            state = torch.from_numpy(state_info['state']).type(torch.FloatTensor).to(device)
            parallel_mask = torch.from_numpy(state_info['parallel_mask']).type(torch.FloatTensor).to(device)
            instruction = state_info['instruction']
            last_level_tiles = state_info['last_level_tiles']
            cur_level = state_info['cur_level']
            action, log_prob, log_prob_mask = actor(state, parallel_mask, instruction, last_level_tiles, cur_level)
            state, state_info, sol, reward, reward_saved, latency, energy, done, info = env.step(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            log_prob_masks.append(log_prob_mask)
            # print(info.sum())
            if done:
                break
        rewards = np.stack(rewards, axis=0)
        log_probs = torch.stack(log_probs, dim=0)
        log_prob_masks = torch.stack(log_prob_masks, dim=0)
        max_len = 0
        print(log_prob_masks.size())
        for s in sol:
            max_len = max(max_len, len(s))
        for ids, s in enumerate(sol):
            if len(s) < max_len:
                log_prob_masks[len(s):] = 0
        policy_loss += compute_policy_loss(rewards, log_probs, log_prob_masks)
        best_idx = np.argmax(reward_saved)
        if reward_saved[best_idx] > best_reward:
            best_reward = reward_saved[best_idx]
            agent_chkpt['best_actor'] = actor.state_dict()
            agent_chkpt['best_reward'] = best_reward
            agent_chkpt['best_latency'] = latency[best_idx]
            agent_chkpt['best_energy'] = energy[best_idx]
            agent_chkpt['best_sol'] = sol[best_idx]
            agent_chkpt['best_state'] = state[best_idx]
            print("Epoch {}, Best Reward: {}, Best Sol: {}".format(ep, best_reward, sol[best_idx]))

        agent_chkpt['best_reward_record'].append(agent_chkpt['best_reward'])
        agent_chkpt['best_latency_record'].append(agent_chkpt['best_latency'])
        agent_chkpt['best_energy_record'].append(agent_chkpt['best_energy'])
        agent_chkpt['best_sols'].append(agent_chkpt['best_sol'])
        log_str = "Epoch {},  Best Reward: {}, Best Sol: {}\n".format(ep, best_reward, agent_chkpt['best_sol'])
        print(log_str)
        epf.write(log_str)
        epf.flush()

        policy_loss /= num_episodes
        actor_optimizer.zero_grad()
        policy_loss.backward()
        # torch.nn.utils.clip_grad_norm_(actor.parameters(), CLIPPING_MODEL)
        actor_optimizer.step()

    return agent_chkpt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--fitness', type=str, default="latency", help='objective fitness')
    parser.add_argument('--parRS', default=False, action='store_true', help='Parallize across R S dimension')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs')
    parser.add_argument('--outdir', type=str, default="outdir", help='output directiory')
    parser.add_argument('--num_pe', type=int, default=168, help='number of PEs')
    parser.add_argument('--l1_size', type=int, default=512, help='L1 size')
    parser.add_argument('--l2_size', type=int, default=108000, help='L2 size')
    parser.add_argument('--model', type=str, default="vgg16", help='Model to run')
    parser.add_argument('--slevel_min', type=int, default=2, help='parallelization level min')
    parser.add_argument('--slevel_max', type=int, default=2, help='parallelization level max')
    parser.add_argument('--log_level', type=int, default=1, help='Detail: 2, runtimeinfo: 1')
    parser.add_argument('--seed', type=int, default=42)

    opt = parser.parse_args()
    m_file_path = "../../data/model/"

    m_file = os.path.join(m_file_path, opt.model + ".csv")
    df = pd.read_csv(m_file)
    model_defs = df.to_numpy()

    num_layers, dim_size = model_defs.shape
    dim2chkpt = {}
    for i in range(0, 1):
        dimension = model_defs[i]
        dimension = np.array([64,128,392,392,2,2,2,4])
        outdir = opt.outdir
        outdir = os.path.join("../../../", outdir)
        exp_name = "{}_SL-{}-{}_F-{}_PE-{}_L1-{}_L2-{}_EPOCH-{}/layer-{}".format(opt.model,opt.slevel_min, opt.slevel_max,opt.fitness, opt.num_pe, opt.l1_size, opt.l2_size, opt.epochs, i)

        outdir_exp = os.path.join(outdir, exp_name)
        os.makedirs(outdir, exist_ok=True)
        os.makedirs(outdir_exp, exist_ok=True)

        dimension_to_key = ','.join(str(j) for j in dimension)
        if dimension_to_key in dim2chkpt:
            agent_chkpt = dim2chkpt[dimension_to_key]
            torch.save(agent_chkpt, os.path.join(outdir_exp, 'agent_chkpt.plt'))
            print("repeated")
        else:
            chkpt_file_t = "{}".format("result")
            log_file = os.path.join(outdir_exp, chkpt_file_t + ".log")
            epf = open(log_file, 'a')
            print(dimension, dimension_to_key)
            try:
                set_seed(opt.seed)
                agent_chkpt = train(dimension)
                dim2chkpt[dimension_to_key] = agent_chkpt
                torch.save(agent_chkpt, os.path.join(outdir_exp, 'agent_chkpt.plt'))
            finally:
                for f in glob.glob("*.m"):
                    os.remove(f)
                for f in glob.glob("*.csv"):
                    os.remove(f)
