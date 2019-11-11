#!/home/pzhang/anaconda3/bin/python3.5

import os
import numpy as np
import argparse
from dataset.kaldi_io import open_or_fd, read_mat_ark, write_vec_flt
from misc.utils import read_config
import sys
from model.trainer import Trainer


parser = argparse.ArgumentParser()
parser.add_argument("-n", "--normalize", action="store_true", help="Normalize the embedding before averaging and output.")
parser.add_argument("--config", type=str, help="The configuration file.")
parser.add_argument("model_dir", type=str, help="The model directory.")
parser.add_argument("rspecifier", type=str, help="Kaldi feature rspecifier (or ark file).")
parser.add_argument("wspecifier", type=str, help="Kaldi output wspecifier (or ark file).")


if __name__ == "__main__":

    args = parser.parse_args()
    params = read_config(args.config)
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    nnet_dir = os.path.join(args.model_dir, "nnet")

    with open(os.path.join(nnet_dir, "num_speakers"), 'r') as f:
        num_speakers = f.readline().strip()


    trainer = Trainer(params=params, model_dir=args.model_dir, num_speakers=int(num_speakers))
    load_model_dir = os.path.join(args.model_dir, "checkpoint")
    trainer.load(model=trainer.network, model_name=os.path.join(load_model_dir, "net.pth"))
    trainer.network.eval()

    if args.rspecifier.rsplit(".", 1)[1] == "scp":
        sys.exit("The rspecifier must be ark or input pipe")

    fp_out = open_or_fd(args.wspecifier, "wb")
    for index, (key, feature) in enumerate(read_mat_ark(args.rspecifier)):
        feature = trainer.test_transform(feature)
        _, embedding = trainer.network(feature)
        embedding = embedding.squeeze(0).cpu().detach().numpy()
        if args.normalize:
            embedding /= np.sqrt(np.sum(np.square(embedding)))
        write_vec_flt(fp_out, embedding, key=key)
    fp_out.close()
    trainer.close()