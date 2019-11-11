#!/home/pzhang/anaconda3/bin/python3.5

import torch
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import random
import argparse
import os
from misc.utils import save_codes_and_config
from dataset.kaldi_io import FeatureReader
from dataset.data_loader import KaldiDataRandomQueue
from model.trainer import Trainer


parser = argparse.ArgumentParser()
parser.add_argument("-c", "--continue-training", action="store_true", help="About whether to continue training.")
parser.add_argument("--config", type=str, help="The configuration file.")
parser.add_argument("train_dir", type=str, help="The data directory of the training set.")
parser.add_argument("train_spklist", type=str, help="The spklist file maps the TRAINING speakers to the indices.")
parser.add_argument("model", type=str, help="The output model directory.")

if __name__ == '__main__':
    args = parser.parse_args()

    params = save_codes_and_config(args.continue_training, args.model, args.config)

    model_dir = os.path.join(args.model, "nnet")

    os.environ['CUDA_VISIBLE_DEVICES'] = params.gpu_id

    torch.manual_seed(params.random_seed)
    np.random.seed(params.random_seed)
    random.seed(params.random_seed)

    dim = FeatureReader(args.train_dir).get_dim()
    with open(os.path.join(model_dir, "feature_dim"), 'w') as f:
        f.write("%d\n" % dim)

    num_total_train_speakers = KaldiDataRandomQueue(args.train_dir, args.train_spklist).num_total_speakers  # 训练说话人数目
    with open(os.path.join(model_dir, "num_speakers"), 'w') as f:
        f.write("%d\n" % num_total_train_speakers)
    
    trainer = Trainer(params, args.model, num_total_train_speakers)
    trainer.build(loss_type=params.loss_func)

    if args.continue_training:
        checkpoint = torch.load(os.path.join(trainer.model, 'net.pth'))
        start_epoch = checkpoint['epoch'] + 1
        trainer.optimizer.load_state_dict(checkpoint['optimizer'])
        trainer.network.load_state_dict(checkpoint['state_dict'])
    else:
        start_epoch = 0

    learning_scheduler = lr_scheduler.StepLR(trainer.optimizer, step_size=params.reduce_lr_epochs, gamma=params.reduce_lr_gamma)
    for epoch in range(start_epoch, params.num_epochs):
        trainer.train(epoch=epoch, data=args.train_dir, spklist=args.train_spklist)
        learning_scheduler.step()
    trainer.close()
