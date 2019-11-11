#!/bin/bash

cmd="run.pl"
normalize=false
nj=30
stage=0

echo "$0 $@" 

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
  echo "Usage: $0 [options] <config> <nnet-dir> <data> <embeddings-dir>"
  echo "Options:"
  echo "  --normalize <false>"
  echo ""
  exit 100
fi

config=$1
nnet_dir=$2
data=$3
dir=$4

export PYTHONPATH=`pwd`/../../

for f in $data/feats.scp $data/vad.scp;  do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

mkdir -p $dir/log

utils/split_data.sh $data $nj
echo "$0: extracting embeddings for $data"
sdata=$data/split$nj/JOB

feat="ark:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:${sdata}/feats.scp ark:- | select-voiced-frames ark:- scp,s,cs:${data}/vad.scp ark:- |"

if [ $stage -le 0 ]; then
  echo "$0: extracting xvectors from nnet"
  $cmd JOB=1:$nj ${dir}/log/extract.JOB.log \
    nnet/wrap/extract_wrapper.sh --normalize $normalize $config \
    $nnet_dir "$feat" "ark:| copy-vector ark:- ark,scp:${dir}/xvector.JOB.ark,${dir}/xvector.JOB.scp"
fi

if [ $stage -le 1 ]; then
  echo "$0: combining xvectors across jobs"
  for j in $(seq $nj); do cat $dir/xvector.$j.scp; done >$dir/xvector.scp || exit 1;
fi

if [ $stage -le 2 ]; then
  # Average the utterance-level xvectors to get speaker-level xvectors
  echo "$0: computing mean of xvectors for each speaker"
  if $normalize; then
    echo "$0:   Normalize xvectors before computing the mean."
    $cmd $dir/log/speaker_mean.log \
      ivector-normalize-length --scaleup=false scp:$dir/xvector.scp ark:- \| \
      ivector-mean ark:$data/spk2utt ark:- ark:- ark,t:$dir/num_utts.ark \| \
      ivector-normalize-length --scaleup=false ark:- ark,scp:$dir/spk_xvector.ark,$dir/spk_xvector.scp || exit 1
  else
    $cmd $dir/log/speaker_mean.log \
      ivector-mean ark:$data/spk2utt scp:$dir/xvector.scp \
        ark,scp:$dir/spk_xvector.ark,$dir/spk_xvector.scp ark,t:$dir/num_utts.ark || exit 1;
  fi
fi

if [ $stage -le 3 ]; then
  if $normalize; then
    # Normalize the output embeddings
    cp $dir/xvector.scp $dir/xvector_before_norm.scp
    $cmd $dir/log/length_norm.log \
      ivector-normalize-length --scaleup=false scp:$dir/xvector_before_norm.scp ark,scp:$dir/xvector.ark,$dir/xvector.scp
  fi
fi



