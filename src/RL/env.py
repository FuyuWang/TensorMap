import numpy as np
import copy, random
import os
from subprocess import Popen, PIPE
from multiprocessing.pool import Pool
from multiprocessing import cpu_count
import pandas as pd
import math
from collections import defaultdict, OrderedDict

m_type_dicts = {1:"CONV", 2:"DSCONV", 3:"CONV", 4:"TRCONV"}


class MaestroEnvironment(object):
    def __init__(self, dimension, fitness="latency", par_RS=False, num_pe=64, l1_size=512, l2_size=108000, NocBW=81920000,
                 slevel_min=2, slevel_max=2, log_level=2, batch_size=10):
        super(MaestroEnvironment,self).__init__()
        self.dimension = dimension
        self.dim_max = np.max(dimension)
        self.dimension_dict = {"K":dimension[0], "C":dimension[1], "Y":dimension[2], "X":dimension[3], "R":dimension[4],"S":dimension[5], "T":dimension[6]}
        self.lastcluster_dict = {"K":dimension[0], "C":dimension[1], "Y":dimension[2], "X":dimension[3], "R":dimension[4],"S":dimension[5], "T":dimension[6]}
        # dst_path = "../../cost_model/maestro"
        dst_path = "../../cost_model/maestro"

        maestro = dst_path
        self._executable = "{}".format(maestro)
        self.out_repr = set(["K", "C", "R", "S"])
        self.num_pe = num_pe
        self.fitness = fitness
        self.cluster_space = ["K", "C", "Y","X","R","S"] if par_RS else ["K", "C", "Y","X"]
        self.dim2id = {"K":1, "C":2, "Y":3, "X":4, "R":5, "S":6}
        self.id2dim = {1:"K", 2:"C", 3:"Y", 4:"X", 5:"R", 6:"S"}
        self.l1_size = l1_size
        self.l2_size = l2_size
        self.NocBW = NocBW
        self.slevel_min = slevel_min
        self.slevel_max = slevel_max
        self.log_level = log_level
        self.KCscale = 2
        self.YXscale = 2

        self.cur_level = 1
        self.total_levels = self.slevel_min
        self.best_reward = float('-inf')
        self.min_reward = float('inf')
        self.mode = 0
        self.mode_sequence = [2, 3, 4, 5, 6, 7]
        self.total_eps_reward = None
        self.last_reward = 0.
        self.level_steps = 8
        self.parallel_mask = None
        # self.remaining_budgets = None

        self.batch_size = batch_size
        self.max_tile2_size = 64
        self.max_tile_size = 64000

        self.last_level_tiles = np.zeros((self.batch_size, 6), dtype=np.int32)
        self.last_level_tiles[:, 0:] = np.maximum(0, np.ceil(np.log2(self.dimension[0:6])) * 2 - 1)
        print(self.last_level_tiles)

        self.action_space = np.ones((self.batch_size, 30), dtype=np.int32)
        for i in range(1, 30):
            self.action_space[:, i] = pow(2, i // 2) + pow(2, (i - 1) // 2)
        print(self.action_space[0])

        # self.dimension_prime = {}
        # self.prime2idx = {}
        # primes = set()
        # for idx, key in self.id2dim.items():
        #     tile_budget = self.get_prime_factors(min(self.dimension[idx-1], 128))
        #     # tile_budget = self.get_prime_factors(self.dimension[idx-1])
        #     self.dimension_prime[key] = tile_budget
        #     for k in tile_budget.keys():
        #         primes.add(k)
        # primes = sorted(primes)
        # self.prime2idx = {pf: i for i, pf in enumerate(primes)}
        # self.num_primes = len(self.prime2idx.keys())
        # print(self.dimension_prime, self.prime2idx)

        # self.tile_budgets = np.zeros((self.batch_size, len(self.dimension) - 1, self.num_primes), dtype=np.int32)
        # for i, key in self.id2dim.items():
        #     tile_budget = self.dimension_prime[key]
        #     for k, v in self.prime2idx.items():
        #         self.tile_budgets[:, i - 1, v] = tile_budget[k]

        self.state = np.ones((self.batch_size, self.slevel_max * (7 + 1)), dtype=np.int32)
        min_dim = np.argmin(self.dimension[1:4]) + 2
        if self.slevel_max == 2:
            self.state[0:self.batch_size, :] = np.array([min_dim, 1, 1, 1, 1, 1, 1, 1, min_dim, 1, 1, 1, 1, 1, 1, 1],
                                                        dtype=np.int32)
        else:
            self.state[0:self.batch_size, :] = np.array(
                [min_dim, 1, 1, 1, 1, 1, 1, 1, min_dim, 1, 1, 1, 1, 1, 1, 1, min_dim, 0, 1, 1, 1, 1, 1, 1], dtype=np.int32)

        pool = Pool(min(self.batch_size, cpu_count()))
        return_list = pool.map(self.get_reward, self.state)
        self.initial_reward = np.zeros(self.batch_size)
        for i in range(self.batch_size):
            self.initial_reward[i] = return_list[i][1]
        self.min_reward = self.initial_reward.min()
        self.info = np.ones(self.batch_size)
        print(self.min_reward)

    def get_prime_factors(self, n):
        primes = defaultdict(int)
        while n % 2 == 0:
            primes['2'] += 1
            n = n // 2
        for i in range(3, int(math.sqrt(n)) + 1, 2):
            while n % i == 0:
                primes[f'{i}'] += 1
                n = n // i
        if n > 2:
            primes[f'{n}'] += 1
        return primes

    def epoch_reset(self, dimension, fitness):
        self.cur_level = 1
        self.total_levels = self.slevel_min
        self.state = np.ones((self.batch_size, self.slevel_max*(7+1)), dtype=np.int32)
        min_dim = np.argmin(self.dimension[1:4]) + 2
        if self.slevel_max == 2:
            self.state[0:self.batch_size, :] = np.array([min_dim, 1, 1, 1, 1, 1, 1, 1, min_dim, 1, 1, 1, 1, 1, 1, 1],
                                                        dtype=np.int32)
        else:
            self.state[0:self.batch_size, :] = np.array(
                [min_dim, 1, 1, 1, 1, 1, 1, 1, min_dim, 1, 1, 1, 1, 1, 1, 1, min_dim, 0, 1, 1, 1, 1, 1, 1],
                dtype=np.int32)
        self.mode = 0
        self.mode_sequence = [2, 3, 4, 5, 6, 7]
        # random.shuffle(self.mode_sequence)
        self.mode_sequence.insert(0, 1)
        self.mode_sequence.insert(0, 0)

        self.last_reward = copy.deepcopy(self.initial_reward)
        self.best_reward = float('-inf')
        self.dimension = dimension
        self.fitness = fitness
        self.dim_max = np.max(dimension)

        self.parallel_mask = np.zeros((self.batch_size, 4))
        if self.dimension[0] <= 3 or self.dimension[1] <= 3:
            if self.dimension[0] <= 3:
                self.parallel_mask[:, 0] = float("-inf")
            else:
                self.parallel_mask[:, 1] = float("-inf")

        self.last_level_tiles = np.zeros((self.batch_size, 6), dtype=np.int32)
        # if self.dimension[0] <= 3:
        #     self.last_level_tiles[:, 0] = self.dimension[0]
        # else:
        #     self.last_level_tiles[:, 0] = self.dimension[0] // self.KCscale + 1
        self.last_level_tiles[:, 0:] = np.maximum(0, np.ceil(np.log2(self.dimension[0:6])) * 2 - 1)

        # if self.dimension[1] <= 3:
        #     self.last_level_tiles[:, 1] = self.dimension[1]
        # else:
        #     self.last_level_tiles[:, 1] = self.dimension[1] // self.KCscale + 1
        #
        # self.last_level_tiles[:, 2] = self.dimension[2] // self.YXscale + 1
        # self.last_level_tiles[:, 3] = self.dimension[3] // self.YXscale + 1
        # self.last_level_tiles[:, 4] = self.dimension[4]
        # self.last_level_tiles[:, 5] = self.dimension[5]
        # print(self.last_level_tiles)

        next_state = np.zeros((self.batch_size, self.slevel_max * self.level_steps))
        next_state[:, 0] = 1 / (self.slevel_max * self.level_steps)
        next_state[:, 1] = self.state[:, 0] / 4
        next_state[:, 2:8] = self.state[:, 2:8] / np.minimum(self.dimension[0:6], self.max_tile_size)
        next_state[:, 8] = self.state[:, 8] / 4
        next_state[:, 9] = self.state[:, 9] / min(self.dim_max, self.max_tile_size)
        next_state[:, 10:16] = self.state[:, 10:16] / np.minimum(self.dimension[0:6], self.max_tile_size)
        if self.slevel_max > 2:
            next_state[:, 16] = self.state[:, 16] / 4
            next_state[:, 17] = self.state[:, 17] / min(self.dim_max, self.max_tile_size)
            next_state[:, 18:] = self.state[:, 18:] / np.minimum(self.dimension[0:6], self.max_tile_size)
        # next_state = (np.sqrt(next_state) - 0.5) * 2
        next_state = np.sqrt(next_state)
        state_info = {}
        state_info['state'] = next_state
        state_info['parallel_mask'] = self.parallel_mask
        state_info['instruction'] = self.mode_sequence[self.mode]
        state_info['last_level_tiles'] = self.last_level_tiles
        state_info['cur_level'] = self.cur_level
        self.info = np.ones(self.batch_size)
        return state_info

    def get_out_repr(self, x):
        if x in self.out_repr:
            return x
        else:
            return x + "'"

    def get_reward(self, state):
        sol = []
        for i in range(self.slevel_max):
            if state[i * self.level_steps + 1] <= 0:
                break
            sol.append([self.id2dim[state[i * self.level_steps]], state[i * self.level_steps + 1]])
            for j in range(1, 7):
                sol.append([self.id2dim[j], state[i * self.level_steps + j + 1]])
        reward, latency, energy, l1_size, l2_size = self.oberserve_maestro(sol)
        constraint = (l1_size, l2_size)
        return sol, reward, latency, energy, constraint

    def step(self, action):
        state_info = {}
        done = 0
        if self.mode_sequence[self.mode] == 0:
            parallel_action = action.cpu().numpy()[:, 0]
            stop_action = action.cpu().numpy()[:, 1]

            if self.cur_level == 1:
                self.state[:, (self.cur_level - 1) * self.level_steps] = parallel_action + 1
                cluster_size = self.dimension[self.state[:, (self.cur_level - 1) * self.level_steps] - 1]
                scale = np.zeros(self.batch_size)
                scale[parallel_action<=1] = self.KCscale
                scale[parallel_action>1] = self.YXscale
                self.last_level_tiles[np.arange(self.batch_size), parallel_action] = np.minimum(
                    self.last_level_tiles[np.arange(self.batch_size), parallel_action],
                    np.maximum(0, np.ceil(np.log2(cluster_size / self.num_pe)) * 2 - 1))
                    # np.ceil(cluster_size / self.num_pe) // scale + 1)
                cluster_size = np.minimum(cluster_size, self.num_pe)
                # if self.dimension[0] <= 3:
                #     self.parallel_mask = np.array([float("-inf"), 0, 0., 0.])
                # else:
                #     self.parallel_mask = np.array([0., 0, 0., 0.])
                # print(self.level, self.mode, parallel_action,self.state)
            else:
                self.state[:, (self.cur_level - 1) * self.level_steps] = parallel_action + 1
                # print(self.level, self.mode, parallel_action, self.state)
                par_dim = self.state[:, (self.cur_level - 1) * self.level_steps]

                last_tiles = self.state[
                    np.arange(self.batch_size), (self.cur_level - 2) * self.level_steps + par_dim + 1]

                last_cluster_size = self.state[:, (self.cur_level - 2) * self.level_steps + 1]

                cluster_size = np.minimum(last_tiles, self.num_pe // last_cluster_size)

                scale = np.zeros(self.batch_size)
                scale[parallel_action <= 1] = self.KCscale
                scale[parallel_action > 1] = self.YXscale
                self.last_level_tiles[np.arange(self.batch_size), parallel_action] = np.minimum(
                    self.last_level_tiles[np.arange(self.batch_size), parallel_action],
                    np.maximum(0, np.ceil(np.log2(last_tiles / cluster_size)) * 2 - 1))
                    # np.ceil(last_tiles / cluster_size) // scale + 1)

            cluster_size[stop_action == 1] = 0

            self.state[:, (self.cur_level - 1) * self.level_steps + 1] = cluster_size

            # print(self.state[:, (self.cur_level - 1) * self.level_steps + 2:(self.cur_level - 1) * self.level_steps + 8].shape)
            self.state[:, (self.cur_level - 1) * self.level_steps + 2:(self.cur_level - 1) * self.level_steps + 8] = np.minimum(1, self.state[:, (self.cur_level - 1) * self.level_steps + 2:(self.cur_level - 1) * self.level_steps + 8] + np.expand_dims(1 - stop_action, axis=-1))

            self.parallel_mask[np.arange(self.batch_size), parallel_action] = float('-inf')
            self.mode += 1
        else:
            tile_action = action.cpu().numpy()[:, 0]
            # tile_size = 1
            # for k, v in self.prime2idx.items():
            #     tile_size *= pow(int(k), tile_action[:, v])

            # if self.mode_sequence[self.mode] == 2:
            #     tile_size = self.KCscale * tile_action
            #     tile_size[tile_action == 0] = 1
            # elif self.mode_sequence[self.mode] == 3:
            #     if self.dimension[1] == 3:
            #         tile_size = tile_action + 1
            #     else:
            #         tile_size = self.KCscale * tile_action
            #     tile_size[tile_action == 0] = 1
            # elif self.mode_sequence[self.mode] == 4:
            #     tile_size = self.YXscale * tile_action
            #     tile_size[tile_action == 0] = 1
            # elif self.mode_sequence[self.mode] == 5:
            #     tile_size = self.YXscale * tile_action
            #     tile_size[tile_action == 0] = 1
            # else:
            #     tile_size = tile_action + 1
            tile_size = self.action_space[np.arange(self.batch_size), tile_action]
            tile_size = np.minimum(tile_size, self.dimension[self.mode_sequence[self.mode] - 2])
            self.state[:, (self.cur_level-1)*self.level_steps+self.mode_sequence[self.mode]] *= tile_size
            self.last_level_tiles[:, self.mode_sequence[self.mode] - 2] = tile_action

        next_state = np.zeros((self.batch_size, self.slevel_max * self.level_steps))
        next_state[:, 0] = 1 / (self.slevel_max * self.level_steps)
        next_state[:, 1] = self.state[:, 0] / 4
        next_state[:, 2:8] = self.state[:, 2:8] / np.minimum(self.dimension[0:6], self.max_tile_size)
        next_state[:, 8] = self.state[:, 8] / 4
        next_state[:, 9] = self.state[:, 9] / min(self.dim_max, self.max_tile_size)
        next_state[:, 10:16] = self.state[:, 10:16] / np.minimum(self.dimension[0:6], self.max_tile_size)
        if self.slevel_max > 2:
            next_state[:, 16] = self.state[:, 16] / 4
            next_state[:, 17] = self.state[:, 17] / min(self.dim_max, self.max_tile_size)
            next_state[:, 18:] = self.state[:, 18:] / np.minimum(self.dimension[0:6], self.max_tile_size)
        # next_state = (np.sqrt(next_state) - 0.5) * 2
        next_state = np.sqrt(next_state)

        pool = Pool(min(self.batch_size, cpu_count()))
        return_list = pool.map(self.get_reward, self.state)
        sol = []
        reward = np.zeros(self.batch_size)
        latency = np.zeros(self.batch_size)
        energy = np.zeros(self.batch_size)
        for i in range(self.batch_size):
            sol.append(return_list[i][0])
            reward[i] = return_list[i][1]
            latency[i] = return_list[i][2]
            energy[i] = return_list[i][3]

        self.info[reward == float('-inf')] = 0
        reward_saved = copy.deepcopy(reward)
        reward_saved[reward_saved == float('-inf')] = float('inf')
        self.min_reward = min(self.min_reward, reward_saved.min())
        reward_saved[reward_saved == float('inf')] = self.min_reward * 2
        reward = reward_saved - self.last_reward
        self.last_reward = reward_saved

        self.mode += 1
        if self.mode == self.level_steps:
            if self.cur_level == self.slevel_max and not done:
                done = 1
            self.cur_level += 1
            self.mode = 0
        state_info['state'] = next_state
        state_info['parallel_mask'] = self.parallel_mask
        state_info['instruction'] = self.mode_sequence[self.mode]
        state_info['last_level_tiles'] = self.last_level_tiles
        state_info['cur_level'] = self.cur_level
        # print(self.mode, self.last_level_tiles[0, self.mode-2])
        return self.state, state_info, sol, reward, reward_saved, latency, energy, done, self.info

    def write_maestro(self, indv, layer_id=0, m_file=None):
        m_type = m_type_dicts[int(self.dimension[-1])]
        stride = int(self.dimension[-2])
        with open("{}.m".format(m_file), "w") as fo:
            fo.write("Network {} {{\n".format(layer_id))
            fo.write("Layer {} {{\n".format(m_type))
            fo.write("Type: {}\n".format(m_type))
            # fo.write("Stride { X: {}, Y: {} }\n".format(stride, stride))
            fo.write("Stride { X: " + str(stride) + ", Y: " + str(stride) + " }\n")
            dim = self.dimension[0:6]
            # dim[2] *= stride
            # dim[3] *= stride
            fo.write("Dimensions {{ K: {:.0f}, C: {:.0f}, Y: {:.0f}, X: {:.0f}, R: {:.0f}, S: {:.0f} }}\n".format(*dim))
            fo.write("Dataflow {\n")
            for k in range(0, len(indv), 7):
                for i in range(k, k + 7):
                    d, d_sz = indv[i]
                    if i % 7 == 0:
                        if k != 0:
                            fo.write("Cluster({},P);\n".format(d_sz))
                    else:
                        sp = "SpatialMap" if d == indv[k][0] else "TemporalMap"
                        if not (m_type == "DSCONV" and self.get_out_repr(d) == "K"):
                            if self.get_out_repr(d) == "Y'" or self.get_out_repr(d) == "X'":
                                fo.write("{}({},{}) {};\n".format(sp, d_sz, d_sz, self.get_out_repr(d)))
                                # if sp == "SpatialMap":
                                #     fo.write("{}({},{}) {};\n".format(sp, d_sz, d_sz, self.get_out_repr(d)))
                                # else:
                                #     fo.write("{}({},{}) {};\n".format(sp, d_sz, 1, self.get_out_repr(d)))

                            else:
                                fo.write("{}({},{}) {};\n".format(sp, d_sz, d_sz, self.get_out_repr(d)))
            fo.write("}\n")
            fo.write("}\n")
            fo.write("}")

    def oberserve_maestro(self, indv):
        m_file = "{}".format(random.randint(0, 2**32))
        self.write_maestro(indv,m_file=m_file)
        os.remove("./{}.csv".format(m_file)) if os.path.exists("./{}.csv".format(m_file)) else None

        # command = [self._executable,
        #            "--Mapping_file={}.m".format(m_file),
        #            "--full_buffer=false", "--noc_bw=81920000",
        #            "--noc_hops=1", "--noc_hop_latency=1",
        #            "--noc_mc_support=true", "--num_pes={}".format(self.num_pe),
        #            "--num_simd_lanes=1", "--l1_size=81920000",
        #            "--l2_size=81920000", "--print_res=false", "--print_res_csv_file=true",
        #            "--print_log_file=false", "--print_design_space=false", "--msg_print_lv=0"]
        command = [self._executable,
                   "--Mapping_file={}.m".format(m_file),
                   "--full_buffer=false", "--noc_bw={}".format(self.NocBW),
                   "--noc_hops=1", "--noc_hop_latency=1",
                   "--noc_mc_support=true", "--num_pes={}".format(self.num_pe),
                   "--num_simd_lanes=1", "--l1_size={}".format(self.l1_size),
                   "--l2_size={}".format(self.l2_size), "--print_res=false", "--print_res_csv_file=true",
                   "--print_log_file=false", "--print_design_space=false", "--msg_print_lv=0"]
        # command = [self._executable,
        #            "--Mapping_file={}.m".format(m_file),
        #            "--full_buffer=false", "--noc_bw=81920000",
        #            "--noc_hops=1", "--noc_hop_latency=1",
        #            "--noc_mc_support=true", "--num_pes={}".format(self.num_pe),
        #            "--num_simd_lanes=1", "--l1_size={}".format(self.l1_size),
        #            "--l2_size={}".format(self.l2_size), "--print_res=false", "--print_res_csv_file=true",
        #            "--print_log_file=false", "--print_design_space=false", "--msg_print_lv=0"]

        process = Popen(command, stdout=PIPE, stderr=PIPE)
        stdout, stderr = process.communicate()
        process.wait()
        os.remove("./{}.m".format(m_file)) if os.path.exists("./{}.m".format(m_file)) else None
        try:
            df = pd.read_csv("./{}.csv".format(m_file))
            layer_name = df[" Layer Number"]
            runtime = np.array(df[" Runtime (Cycles)"]).reshape(-1, 1)
            throughput = np.array(df[" Throughput (MACs/Cycle)"]).reshape(-1, 1)
            energy = np.array(df[" Activity count-based Energy (nJ)"]).reshape(-1, 1)
            area = np.array(df[" Area"]).reshape(-1, 1)
            power = np.array(df[" Power"]).reshape(-1, 1)
            l1_size = np.array(df[" L1 SRAM Size (Bytes)"]).reshape(-1, 1)
            l2_size = np.array(df["  L2 SRAM Size (Bytes)"]).reshape(-1, 1)
            mac = np.array(df[" Num MACs"]).reshape(-1, 1)
            os.remove("./{}.csv".format(m_file))  if os.path.exists("./{}.csv".format(m_file)) else None
            os.remove("./log.txt") if os.path.exists("./log.txt") else None
            self.observation = [np.mean(x) for x in [runtime, throughput, energy, area, l1_size, l2_size, mac, power]]

            def catch_exception():
                if l1_size > self.l1_size or l2_size > self.l2_size or runtime < 1 or l1_size < 0 or l2_size < 0:
                    return True
                else:
                    return False

            if len(str(stdout)) > 3 or catch_exception():
                return float('-inf'), float('-inf'), float('-inf'), np.sum(l1_size), np.sum(l2_size)
            return self.judge()
        except Exception as e:
            # print(e, indv)
            return float('-inf'), float('-inf'), float('-inf'), -1, -1

    def judge(self):
        runtime, throughput, energy, area, l1_size, l2_size, mac, power = self.observation
        values = []
        for term in [self.fitness]:
            if term == "energy":
                reward = -energy
            elif term == "thrpt_ave":
                reward = throughput
            elif term == "EDP":
                reward = -energy * runtime * 1E-6
            elif term == "LAP":
                reward = -area * runtime
            elif term == "EAP":
                reward = -area * energy
            elif term == "thrpt" or term == "thrpt_naive":
                reward = throughput
            elif term == "thrpt_btnk":
                reward = throughput
            elif term == "latency":
                reward = -runtime
            elif term == "area":
                reward = -area
            elif term == "l1_size":
                reward = - l1_size
            elif term == "l2_size":
                reward = -l2_size
            elif term == "power":
                reward = -power
            else:
                raise NameError('Undefined fitness type')
            values.append(reward)
        values.append(l1_size)
        values.append(l2_size)
        return reward, -runtime, -energy, l1_size, l2_size
