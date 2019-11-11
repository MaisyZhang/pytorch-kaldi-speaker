#!/bin/bash
# Copyright   2017   Johns Hopkins University (Author: Daniel Garcia-Romero)
#             2017   Johns Hopkins University (Author: Daniel Povey)
#        2017-2018   David Snyder
#             2018   Ewald Enzinger
# Apache 2.0.
#
# See ../README.txt for more info on data required.
# Results (mostly equal error-rates) are inline in comments below.


. ./cmd.sh
. ./path.sh
set -e
voxceleb1_trials=data/voxceleb1_test/trials
ivector_dir=/home/pzhang/mydisk/kaldi_voxceleb/exp/ivector_rst
stage=5
nj=20
num_ubm=2048

if [ $stage -le 1 ]; then 
  # Train the UBM
    sid/train_diag_ubm.sh --cmd "$train_cmd --mem 4G" \
    --nj $nj --num-threads 8 \
    data/train_combined_no_sil $num_ubm \
    ${ivector_dir}/diag_ubm

  sid/train_full_ubm.sh --cmd "$train_cmd --mem 25G" \
    --nj $nj --remove-low-count-gaussians false \
    data/train_combined_no_sil \
    ${ivector_dir}/diag_ubm ${ivector_dir}/full_ubm
fi

if [ $stage -le 2 ]; then
  # utils/subset_data_dir.sh \
  #   --utt-list <(sort -n -k 2 data/train_combined_no_sil/utt2num_frames | tail -n 100000) \
  #   data/train_combined_no_sil data/train_combined_no_sil_100k
  # 用所有数据
  # Train the i-vector extractor.
  sid/train_ivector_extractor.sh --cmd "$train_cmd --mem 16G" \
    --ivector-dim 400 --num-iters 5  \
    ${ivector_dir}/full_ubm/final.ubm data/train_combined_no_sil \
    ${ivector_dir}/extractor
fi

if [ $stage -le 3 ]; then
  sid/extract_ivectors.sh --cmd "$train_cmd --mem 4G" --nj $nj \
    $ivector_dir/extractor data/train_combined_no_sil \
    $ivector_dir/ivectors_train

  sid/extract_ivectors.sh --cmd "$train_cmd --mem 4G" --nj $nj \
    $ivector_dir/extractor data/voxceleb1_test \
    $ivector_dir/ivectors_voxceleb1_test
fi

if [ $stage -le 4 ]; then
  # Compute the mean vector for centering the evaluation i-vectors.
  $train_cmd ${ivector_dir}/ivectors_train/log/compute_mean.log \
    ivector-mean scp:$ivector_dir/ivectors_train/ivector.scp \
    $ivector_dir/ivectors_train/mean.vec || exit 1;
  
  # This script uses LDA to decrease the dimensionality prior to PLDA.
  lda_dim=200
  $train_cmd ${ivector_dir}/ivectors_train/log/lda.log \
    ivector-compute-lda --total-covariance-factor=0.0 --dim=$lda_dim \
    "ark:ivector-subtract-global-mean scp:${ivector_dir}/ivectors_train/ivector.scp ark:- |" \
    ark:data/train_combined_no_sil/utt2spk ${ivector_dir}/ivectors_train/transform.mat || exit 1;

  # Train the PLDA model.
  $train_cmd ${ivector_dir}/ivectors_train/log/plda.log \
    ivector-compute-plda ark:data/train_combined_no_sil/spk2utt \
    "ark:ivector-subtract-global-mean scp:${ivector_dir}/ivectors_train/ivector.scp ark:- | transform-vec ${ivector_dir}/ivectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:-  ark:- |" \
    ${ivector_dir}/ivectors_train/plda || exit 1;
fi

sid/extract_ivectors.sh --cmd "$train_cmd --mem 4G" --nj $nj \
    $ivector_dir/extractor data/voxceleb1_test \
    $ivector_dir/ivectors_voxceleb1_test


if [ $stage -le 5 ]; then
    $train_cmd exp/scores/log/voxceleb1_test_scoring_ivector.log \
    ivector-plda-scoring --normalize-length=true \
    "ivector-copy-plda --smoothing=0.0 ${ivector_dir}/ivectors_train/plda - |" \
    "ark:ivector-subtract-global-mean ${ivector_dir}/ivectors_train/mean.vec scp:${ivector_dir}/ivectors_voxceleb1_test/ivector.scp ark:- | transform-vec ${ivector_dir}/ivectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-subtract-global-mean ${ivector_dir}/ivectors_train/mean.vec scp:${ivector_dir}/ivectors_voxceleb1_test/ivector.scp ark:- | transform-vec ${ivector_dir}/ivectors_train/transform.mat ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    "cat '$voxceleb1_trials' | cut -d\  --fields=1,2 |" exp/scores_voxceleb1_test_ivector || exit 1;
fi

if [ $stage -le 6 ]; then
  eer=`compute-eer <(local/prepare_for_eer.py $voxceleb1_trials exp/scores_voxceleb1_test_ivector) 2> /dev/null`
  mindcf1=`sid/compute_min_dcf.py --p-target 0.01 exp/scores_voxceleb1_test_ivector $voxceleb1_trials 2> /dev/null`
  mindcf2=`sid/compute_min_dcf.py --p-target 0.001 exp/scores_voxceleb1_test_ivector $voxceleb1_trials 2> /dev/null`
  echo "EER: $eer%"
  echo "minDCF(p-target=0.01): $mindcf1"
  echo "minDCF(p-target=0.001): $mindcf2"
  # kaldi
  # EER: 5.329%
  # minDCF(p-target=0.01): 0.4933
  # minDCF(p-target=0.001): 0.6168
  # 我的
  # EER: 7.275%
  # minDCF(p-target=0.01): 0.5758
  # minDCF(p-target=0.001): 0.7850


# 11月1日训练结果 子集训练
# EER: 11.05%
# minDCF(p-target=0.01): 0.7650
# minDCF(p-target=0.001): 0.8996
fi