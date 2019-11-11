#!/home/pzhang/anaconda3/bin/python3.5

from model.tdnn import Tdnn
import os
from model.loss import softmax
import torch
import torch.nn as nn
import collections
import time
import numpy as np
from dataset.data_loader import KaldiDataRandomQueue, KaldiDataSeqQueue, DataOutOfRange


class Trainer(object):
    def __init__(self, params, model_dir, num_speakers):
        self.params = params
        self.num_speakers = num_speakers
        self.network_type = params.network_type
        if params.network_type == 'tdnn':
            self.network = Tdnn(pooling_type=params.pooling_type, num_speakers=num_speakers)
            if params.gpu_num > 1:
                self.network = nn.DataParallel(self.network)
            self.network = self.network.to(torch.device(params.train_type))
        else:
            raise NotImplementedError("Not implement %s network" % params.network_type)

        self.optimizer = None
        self.loss_type = None
        self.loss_network = None

        # The model is saved in model/nnet and the evaluation result is saved in model/nnet/eval
        self.model = os.path.join(model_dir, "nnet")
        self.model_log = os.path.join(model_dir, "log")
        self.checkpoint_dir = os.path.join(model_dir, "checkpoint")
        self.auto_eval_dir = os.path.join(self.model, "eval")

    def load(self, model, model_name):
        model_dict = torch.load(model_name)
        if 'DataParallel' in str(type(model)):
            if not list(model_dict['state_dict'].keys())[0].startswith('module.'):
                new_dict = collections.OrderedDict()
                for key in model_dict['state_dict'].keys():
                    new_dict['module.' + key] = model_dict['state_dict'][key]
                model.load_state_dict(new_dict, strict=True)
            else:
                model.load_state_dict(model_dict['state_dict'], strict=True)
        else:
            if list(model_dict['state_dict'].keys())[0].startswith('module.'):
                new_dict = collections.OrderedDict()
                for key in model_dict['state_dict'].keys():
                    new_dict[key.replace('module.', '')] = model_dict['state_dict'][key]
                model.load_state_dict(new_dict, strict=True)
            else:
                model.load_state_dict(model_dict['state_dict'], strict=True)
        return

    def save(self, epoch, model, optimizer):
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                   '{}/net.pth'.format(self.checkpoint_dir))
        torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                   '{}/net_epoch_{}.pth'.format(self.checkpoint_dir, epoch))

        return

    def close(self):
        print('Training over')
        return

    def transform(self, features, labels):
        labels = torch.Tensor(labels).long().to(torch.device(self.params.train_type))
        features = torch.Tensor(features).to(torch.device(self.params.train_type))
        features = features.permute(0, 2, 1)
        return features, labels

    def test_transform(self, features):
        features = torch.Tensor(features).to(torch.device(self.params.train_type))
        features = features.unsqueeze(0)
        features = features.permute(0, 2, 1)
        return features

    def build(self, loss_type=None, noupdate_var_list=None):
        self.loss_type = loss_type
        if loss_type == 'softmax':
            self.loss_network = softmax().to(torch.device(self.params.train_type))
        else:
            raise NotImplementedError("Not implement %s loss" % self.loss_type)

        if self.params.optimizer == 'sgd':
            opt = torch.optim.SGD(self.network.parameters(), lr=self.params.learning_rate, weight_decay=self.params.weight_l2_regularizer, momentum=self.params.batchnorm_momentum)
        elif self.params.optimizer == 'adam':
            opt = torch.optim.Adam(self.network.parameters(), lr=self.params.learning_rate)
        self.optimizer = opt

    def train(self, epoch, data, spklist, aux_data=None):
        self.network.train()
        curr_step = 0
        data_loader = KaldiDataRandomQueue(data, spklist,
                                           num_parallel=self.params.num_parallel_datasets,
                                           max_qsize=self.params.max_queue_size,
                                           num_speakers=self.params.num_speakers_per_batch,
                                           num_segments=self.params.num_segments_per_speaker,
                                           min_len=self.params.min_segment_len,
                                           max_len=self.params.max_segment_len,
                                           shuffle=True)
        data_loader.start()
        sum_loss, sum_samples = 0, 0
        for step in range(curr_step % self.params.num_steps_per_epoch, self.params.num_steps_per_epoch):
            features, labels = data_loader.fetch()
            sum_samples += len(features)
            features, labels = self.transform(features, labels)
            out, _ = self.network(features)
            torch.cuda.empty_cache()
            loss = self.loss_network(out, labels)
            sum_loss += loss.item() * len(features)
            if step % self.params.show_training_process == 0:
                with open(os.path.join(self.model_log, "iter_loss_log"), 'a') as iter_f:
                    iter_f.write("Time:{}, Epoch:{}, Iter:{}, Loss:{}\n".format(
                        time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                        epoch, step, sum_loss / sum_samples))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            curr_step += 1
        with open(os.path.join(self.model_log, "epoch_loss_log"), 'a') as epoch_f:
            epoch_f.write("Time:{}, Epoch:{}, Loss:{}\n".format(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                                                                epoch, sum_loss / sum_samples))
            self.save(epoch=epoch, model=self.network, optimizer=self.optimizer)
        data_loader.stop()
























