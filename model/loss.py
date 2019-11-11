#!/home/pzhang/anaconda3/bin/python3.5

import torch
import torch.nn as nn


class softmax(nn.Module):
    def __init__(self):
        super(softmax, self).__init__()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        loss = self.ce(logits, labels)

        # 如果有附加的loss计算
        # if 'aux_loss_func' in self.params.dict:
        #     loss_aux = 0
        #     loss += loss_aux

        return loss
