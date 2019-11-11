#!/home/pzhang/anaconda3/bin/python3.5

import torch
import torch.nn as nn
import numpy as np
from model.pooling import statistics_pooling


class Tdnn(nn.Module):
    def __init__(self, pooling_type, num_speakers):
        super(Tdnn, self).__init__()

        if pooling_type == "statistics_pooling":
            self.pooling_layer = statistics_pooling()
        self.num_speakers = num_speakers

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=30, out_channels=512, kernel_size=(5, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(5, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(7, 1)),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.layer4 = nn.Sequential(
            nn.Linear(in_features=512, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )

        self.layer5 = nn.Sequential(
            nn.Linear(in_features=512, out_features=1500),
            nn.BatchNorm1d(1500),
            nn.ReLU()
        )

        self.layer6 = nn.Linear(in_features=3000, out_features=512)
        self.layer6_bn = nn.BatchNorm1d(512)
        self.layer6_act = nn.ReLU()

        self.layer7 = nn.Sequential(
            nn.Linear(in_features=512, out_features=512),
            nn.BatchNorm1d(512),
            nn.ReLU()
        )
        self.softmax_layer = nn.Linear(in_features=512, out_features=num_speakers)

    def forward(self, input):
        x = input.unsqueeze(-1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.permute(0, 2, 1, 3).squeeze(-1)
        b, t, f = x.size()
        x = x.contiguous().view(b * t, f)
        x = self.layer4(x)
        x = self.layer5(x)
        _, f = x.size()
        x = x.contiguous().view(b, t, f)
        x = self.pooling_layer(x)
        x = self.layer6(x)
        embedding = self.layer6_bn(x)
        x = self.layer6_act(embedding)
        x = self.layer7(x)
        out = self.softmax_layer(x)

        return out, embedding


if __name__ == '__main__':
    # Test Network
    batch_size = 64
    frame_length = 300
    feature_dim = 30
    features = torch.rand(batch_size, feature_dim, frame_length)
    model = Tdnn(pooling_type='statistics_pooling', num_speakers=7323)
    out, embedding = model(features)
    print(out)
    print(embedding)








