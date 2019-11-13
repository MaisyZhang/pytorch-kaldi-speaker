#!/bin/bash

# 此脚本用于pair测试集, 对于hp提供的测试集给出的egs, 如果要是其他测试集需要新建脚本

. ./cmd.sh
. ./path.sh
set -e

stage=3
nj=1
root=/home/pzhang/mydisk/hp_test

data=$root/data
exp=$root/exp
mfccdir=$root/mfcc
wav_dir=$root/wav3
nnet_dir=/home/pzhang/mydisk/kaldi_voxceleb/exp/xvector_nnet_1a
output_dir=$exp/xvector_nnet_1a/hp_test
hp_trials=$data/trials


if [ $stage -le 0 ]; then
  # 生成wav.scp, utt2spk, trials
  local/hp_test_data_prep.py $root $wav_dir
  # 生成spk2utt
  utils/utt2spk_to_spk2utt.pl $data/utt2spk > $data/spk2utt
  utils/fix_data_dir.sh $data  # 用来sort一下未规整的数据
fi

if [ $stage -le 1 ]; then
  steps/make_mfcc.sh --write-utt2num-frames true --mfcc-config conf/mfcc.conf --nj $nj --cmd "$train_cmd" \
  $data $exp/make_mfcc $mfccdir
  utils/fix_data_dir.sh $data
  sid/compute_vad_decision.sh --nj $nj --cmd "$train_cmd" \
      $data $exp/make_vad $vaddir
  utils/fix_data_dir.sh $data
fi

if [ $stage -le 2 ]; then
  sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 4G" --nj $nj \
    $nnet_dir $data \
    $output_dir
fi

if [ $stage -le 3 ]; then
  $train_cmd $exp/scores/log/hp_test_scoring.log \
    ivector-plda-scoring --normalize-length=true \
    "ivector-copy-plda --smoothing=0.0 $nnet_dir/xvectors_train/plda - |" \
    "ark:ivector-subtract-global-mean $nnet_dir/xvectors_train/mean.vec scp:$output_dir/xvector.scp ark:- | transform-vec $nnet_dir/xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean $nnet_dir/xvectors_train/mean.vec scp:$output_dir/xvector.scp ark:- | transform-vec $nnet_dir/xvectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$hp_trials' | cut -d\  --fields=1,2 |" $exp/hp_test_score || exit 1;
fi

if [ $stage -le 4 ]; then
  eer=`compute-eer <(local/prepare_for_eer.py $hp_trials $exp/hp_test_score) 2> /dev/null`
  mindcf1=`sid/compute_min_dcf.py --c-miss 10 --p-target 0.01 $exp/hp_test_score $hp_trials 2> /dev/null`
  mindcf2=`sid/compute_min_dcf.py --p-target 0.001 $exp/hp_test_score $hp_trials 2> /dev/null`
  echo "EER: $eer%"
  echo "minDCF(p-target=0.01): $mindcf1"
  echo "minDCF(p-target=0.001): $mindcf2"
fi

exit 0
