#!/home/pzhang/anaconda3/bin/python3.5

import json
import sys
import os
import shutil
import numpy as np
from sklearn import metrics
from scipy.optimize import brentq
from scipy.interpolate import interp1d


class Params():
    def __init__(self, json_path):
        self.update(json_path)

    def update(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        return self.__dict__


class ValidLoss():
    def __init__(self):
        self.min_loss = 1e16
        self.min_loss_epoch = -1


def load_valid_loss(filename):
    min_loss = ValidLoss()
    with open(filename, "r") as f:
        for line in f.readlines():
            epoch, loss = line.strip().split(" ")[:2]
            epoch = int(epoch)
            loss = float(loss)
            if loss < min_loss.min_loss:
                min_loss.min_loss = loss
                min_loss.min_loss_epoch = epoch


def read_config(config):
    params = Params(config)
    return params
    

def save_codes_and_config(continue_training, model, config):
    if continue_training:
        if not os.path.isdir(os.path.join(model, "nnet")) or not os.path.isdir(os.path.join(model, "codes")):
            sys.exit("To continue training the model, nnet and codes must be existed in %s." % model)
            # Simply load the configuration from the saved model.
        print("Continue training from %s." % model)
        params = Params(os.path.join(model, "nnet/config.json"))
    else:
        if os.path.isdir(os.path.join(model, "nnet")):  # 如果已有nnet
            # Backup the codes and configuration in .backup. Keep the model unchanged.
            print("Save backup to %s" % os.path.join(model, ".backup"))
            if os.path.isdir(os.path.join(model, ".backup")):
                print("The dir %s exisits. Delete it and continue." % os.path.join(model, ".backup"))
                shutil.rmtree(os.path.join(model, ".backup"))
            os.makedirs(os.path.join(model, ".backup"))
            if os.path.exists(os.path.join(model, "codes")):
                shutil.move(os.path.join(model, "codes"), os.path.join(model, ".backup/"))
            if os.path.exists(os.path.join(model, "nnet")):
                shutil.move(os.path.join(model, "nnet"), os.path.join(model, ".backup/"))
            if os.path.exists(os.path.join(model, "checkpoint")):
                shutil.move(os.path.join(model, "checkpoint"), os.path.join(model, ".backup/"))
            if os.path.exists(os.path.join(model, "lib")):  # 这个是存什么的
                shutil.move(os.path.join(model, "lib"), os.path.join(model, ".backup/"))

        if os.path.isdir(os.path.join(model, "codes")):
            shutil.rmtree(os.path.join(model, "codes"))
        if os.path.isdir(os.path.join(model, "lib")):
            shutil.rmtree(os.path.join(model, "lib"))
        os.makedirs(os.path.join(model, "codes"))
        os.makedirs(os.path.join(model, "checkpoint"))

        # We need to set the home directory of the tf-kaldi-speaker (TF_KALDI_ROOT).  # 需要设置环境变量
        # if not os.environ.get('TF_KALDI_ROOT'):
        #     tf.logging.error("TF_KALDI_ROOT should be set before training. Refer to path.sh to set the value manually. ")
        #     quit()
        # copy_tree(os.path.join(os.environ['TF_KALDI_ROOT'], "dataset"), os.path.join(model, "codes/dataset/"))
        # copy_tree(os.path.join(os.environ['TF_KALDI_ROOT'], "model"), os.path.join(model, "codes/model/"))
        # copy_tree(os.path.join(os.environ['TF_KALDI_ROOT'], "misc"), os.path.join(model, "codes/misc/"))
        # copy_tree(os.path.join(os.getcwd(), "nnet/lib"), os.path.join(model, "lib"))

        if not os.path.isdir(os.path.join(model, "nnet")):
            os.makedirs(os.path.join(model, "nnet"))
        shutil.copyfile(config, os.path.join(model, "nnet", "config.json"))
        print("Train the model from scratch.")
        params = Params(config)
    return params


def load_lr(filename):
    """Load learning rate from a saved file"""
    learning_rate_array = []
    with open(filename, "r") as f:
        for line in f.readlines():
            _, lr = line.strip().split(" ")
            learning_rate_array.append(float(lr))
    return learning_rate_array


def compute_cos_pairwise_eer(embeddings, labels, max_num_embeddings=1000):
    """Compute pairwise EER using cosine similarity.
    The EER is estimated by interp1d and brentq, so it is not the exact value and may be a little different each time.

    Args:
        embeddings: The embeddings.
        labels: The class labels.
        max_num_embeddings: The max number of embeddings to compute the EER.
    :return: The pairwise EER.
    """
    embeddings /= np.sqrt(np.sum(embeddings ** 2, axis=1, keepdims=True) + 1e-12) # (2240, 512)
    num_embeddings = embeddings.shape[0]  # 2240
    if num_embeddings > max_num_embeddings:
        # Downsample the embeddings and labels
        step = num_embeddings / max_num_embeddings  # 2.24
        embeddings = embeddings[range(0, num_embeddings, step), :]
        labels = labels[range(0, num_embeddings, step)]
        num_embeddings = embeddings.shape[0]

    score_mat = np.dot(embeddings, np.transpose(embeddings))
    scores = np.zeros((num_embeddings * (num_embeddings - 1) / 2))
    keys = np.zeros((num_embeddings * (num_embeddings - 1) / 2))
    index = 0
    for i in range(num_embeddings - 1):
        for j in range(i + 1, num_embeddings):
            scores[index] = score_mat[i, j]
            keys[index] = 1 if labels[i] == labels[j] else 0
            index += 1

    fpr, tpr, thresholds = metrics.roc_curve(keys, scores, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    # thresh = interp1d(fpr, thresholds)(eer)

    with open("test.txt", "w") as f:
        for i in range(num_embeddings):
            if keys[i] == 1:
                f.write("%f target" % scores[i])
            else:
                f.write("%f nontarget" % scores[i])
    return eer
