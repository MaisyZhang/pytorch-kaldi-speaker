#!/home/pzhang/anaconda3/bin/python3.5

import torch


class statistics_pooling:
    def __init__(self):
        super(statistics_pooling, self).__init__()
        self.VAR2STD_EPSILON = 1e-12

    def __call__(self, input):

        mean = torch.mean(input, dim=1, keepdim=True)
        squared_diff = (input - mean) ** 2
        variance = torch.mean(squared_diff, dim=1, keepdim=True)
        mean = mean.squeeze(1)
        variance = variance.squeeze(1)

        mask = (variance[:,:] <= self.VAR2STD_EPSILON).float()
        variance = (1.0 - mask) * variance + mask * self.VAR2STD_EPSILON
        stddev = torch.sqrt(variance)

        stat_pooling = torch.cat((mean, stddev), dim=1)

        return stat_pooling
