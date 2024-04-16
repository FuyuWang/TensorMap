from torch.distributions import Categorical
from torch.distributions import Bernoulli
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def init_weights(m):
    if type(m) == nn.LSTMCell:
        nn.init.orthogonal_(m.weight_hh)
        nn.init.orthogonal_(m.weight_ih)
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0)


class Actor(nn.Module):
    def __init__(self, model_def, slevel_max, slevel_min, par_RS=False,
                 h_size=128, hidden_dim=10, batch_size=10, prime2idx=None):
        super(Actor, self).__init__()

        self.model_def = model_def
        self.h_size = h_size
        self.slevel_max = slevel_max
        self.slevel_min = slevel_min
        dim_length = slevel_max * 8

        self.batch_size = batch_size
        # self.prime2idx = prime2idx
        # self.idx2prime = {value: key for key, value in prime2idx.items()}
        # self.num_primes = len(self.prime2idx.keys())

        self.dim_encoder = nn.Sequential(
            nn.Linear(dim_length, dim_length*hidden_dim),
            nn.ReLU(),
            nn.Linear(dim_length*hidden_dim, h_size),
            nn.ReLU(),
            nn.Linear(h_size, h_size),
            nn.ReLU(),
            nn.Linear(h_size, h_size),
            nn.ReLU(),
        )

        self.cluster_space = 6 if par_RS else 4
        self.parallel_decoder = nn.Sequential(
            nn.Linear(h_size, h_size),
            nn.ReLU(),
            nn.Linear(h_size, self.cluster_space),
        )

        # self.model_def_temp = np.array(np.array([162, 161, 224, 224, 7, 7]))

        self.tile_size = 512
        self.tile_decoder = nn.Sequential(
            nn.Linear(h_size, h_size),
            nn.ReLU(),
            nn.Linear(h_size, self.tile_size),
        )

        self.stop_decoder = nn.Sequential(
            nn.Linear(h_size, h_size),
            nn.ReLU(),
            nn.Linear(h_size, 1),
            nn.Sigmoid()
        )

        self.lstm = torch.nn.LSTMCell(h_size, h_size)

        self.parallel_temperature = 1.
        self.order_temperature = 1.
        self.tile_temperature = 1.
        self.lstm_value = None

        self.init_weight()

    def reset(self):
        self.lstm_value = self.init_hidden()

    def init_weight(self):
        self.apply(init_weights)

    def init_hidden(self):
        weight = next(self.parameters())
        return (weight.new_zeros(self.batch_size, self.h_size),
                weight.new_zeros(self.batch_size, self.h_size))

    def set_tile_temperature(self, temp):
        self.tile_temperature = temp

    def set_order_temperature(self, temp):
        self.order_temperature = temp

    def set_parallel_temperature(self, temp):
        self.parallel_temperature = temp

    def forward(self, state, parallel_mask, instruction, last_level_tiles, cur_level):
        '''
        :param state dim_info if instruction == 0, origin tile state if instruction in [1,6], origin tile and order state if instruction in [7, 12]
        :param instruction int parallel action, order action or tile action
        :param remaining_budgets  next level tile <= last level tile
        :return: parallel dim action
        '''

        dim_feat = self.dim_encoder(state)
        h, x = self.lstm(dim_feat, self.lstm_value)
        self.lstm_value = (h, x)
        if instruction == 0:
            parallel_score = self.parallel_decoder(h) + parallel_mask
            parallel_prob = F.softmax(parallel_score / self.parallel_temperature, dim=1)
            parallel_density = Categorical(parallel_prob)
            parallel_action = parallel_density.sample()
            parallel_log_prob = parallel_density.log_prob(parallel_action)
            parallel_log_prob_mask = ((parallel_mask == 0).sum(dim=-1) > 1).float()

            if cur_level <= self.slevel_min:
                stop_action = parallel_action.new_zeros(self.batch_size)
                stop_log_prob = parallel_score.new_zeros(self.batch_size)
                stop_log_prob_mask = parallel_score.new_zeros(self.batch_size)
            # elif cur_level == self.slevel_max:
            #     stop_action = parallel_action.new_ones(self.batch_size)
            #     stop_log_prob = parallel_score.new_zeros(self.batch_size)
            #     stop_log_prob_mask = parallel_score.new_zeros(self.batch_size)
            else:
                stop_score = self.stop_decoder(h).contiguous().view(self.batch_size)
                stop_density = Bernoulli(stop_score)
                stop_action = stop_density.sample()
                stop_log_prob = stop_density.log_prob(stop_action)
                stop_log_prob_mask = parallel_score.new_ones(self.batch_size)

            action = torch.stack([parallel_action, stop_action.long()], dim=1)
            log_prob = torch.stack([parallel_log_prob, stop_log_prob], dim=1)
            log_prob_mask = torch.stack([parallel_log_prob_mask, stop_log_prob_mask], dim=1)

            return action, log_prob, log_prob_mask
        else:
            last_level_tiles = last_level_tiles[:, instruction - 2]
            tile_mask = torch.zeros(self.batch_size, self.tile_size).to(h.device)
            for i in range(self.batch_size):
                tile_mask[i, last_level_tiles[i]+1:] = float('-inf')
            tile_score = self.tile_decoder(h)
            tile_score = tile_score + tile_mask
            tile_prob = F.softmax(tile_score / self.tile_temperature, dim=1)
            tile_density = Categorical(tile_prob)
            tile_action = tile_density.sample()
            tile_log_prob = tile_density.log_prob(tile_action)
            tile_log_prob_mask = ((tile_mask == 0).sum(dim=-1) > 1).float()

            pad = tile_log_prob.new_zeros(self.batch_size)
            tile_action = torch.stack([tile_action, pad.long()], dim=1)
            tile_log_prob = torch.stack([tile_log_prob, pad], dim=1)
            tile_log_prob_mask = torch.stack([tile_log_prob_mask, pad], dim=1)

            return tile_action, tile_log_prob, tile_log_prob_mask
