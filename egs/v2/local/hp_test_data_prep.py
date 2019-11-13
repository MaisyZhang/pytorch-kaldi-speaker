#!/usr/bin/env python3

import sys
import os
import numpy as np
import glob

def write_wav(wav_dir, test_data_dir):
    pair_dirs = glob.glob(wav_dir + '/*')
    test_wav_scp = test_data_dir + '/wav.scp'
    with open(test_wav_scp, 'w') as f:
        for pair in pair_dirs:
            write_line1 = pair.split('/')[-1] + '-' + '1' + ' ' + pair + '/1.wav' + '\n'
            write_line2 = pair.split('/')[-1] + '-' + '2' + ' ' + pair + '/2.wav' + '\n'
            f.write(write_line1)
            f.write(write_line2)
    pass

def write_utt2spk(wav_dir, test_data_dir):
    pair_dirs = glob.glob(wav_dir + '/*')
    test_utt2spk = test_data_dir + '/utt2spk'
    with open(test_utt2spk, 'w') as f:
        for pair in pair_dirs:
            write_line1 = pair.split('/')[-1] + '-' + '1' + ' ' + pair.split('/')[-1] + '\n'
            write_line2 = pair.split('/')[-1] + '-' + '2' + ' ' + pair.split('/')[-1] + '\n'
            f.write(write_line1)
            f.write(write_line2)
    pass

def write_trials(wav_dir, test_data_dir):
    pair_dirs = glob.glob(wav_dir + '/*')
    test_trials = test_data_dir + '/trials'
    with open(test_trials, 'w') as f:
        for pair in pair_dirs:
            label = np.load(pair + '/label.npy')
            if label[0] == 0:
                write_line = pair.split('/')[-1] + '-' + '1' + ' ' + pair.split('/')[-1] + '-' + '2' + ' ' + 'nontarget' + '\n'
                f.write(write_line)
            else:
                write_line = pair.split('/')[-1] + '-' + '1' + ' ' + pair.split('/')[-1] + '-' + '2' + ' ' + 'target' + '\n'
                f.write(write_line)
    pass


if __name__ == '__main__':
    root_dir = sys.argv[1]
    wav_dir = sys.argv[2]
    test_data_dir = os.path.join(root_dir, 'data')
    if not os.path.exists(test_data_dir):
        os.makedirs(test_data_dir)
    
    # 生成wav.scp
    # 文件名 路径
    write_wav(wav_dir, test_data_dir)

    # 生成utt2spk
    # 文件名 spk
    write_utt2spk(wav_dir, test_data_dir)

    # 生成trials
    # test_1 test_2 nontarget/target
    write_trials(wav_dir, test_data_dir)
    