#!/bin/bash

cmd="run.pl"
continue_training=false

echo "$0 $@"

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
  echo "Usage: $0 [options] <config> <train-dir> <train-spklist> <nnet>"
  echo "Options:"
  echo "  --continue-training <false>"
  echo "  --gpu-id <gpu-id>"
  exit 100
fi

config=$1
train=$2
train_spklist=$3
nnet_dir=$4

if [ $continue_training == 'true' ]; then 
  cmdopts="-c"
fi

# add the library to the python path.
export PYTHONPATH=`pwd`/../../

mkdir -p $nnet_dir/log 

if [ -d $nnetdir/log ] && [ `ls $nnetdir/log | wc -l` -ge 1 ]; then
  mkdir -p $nnetdir/.backup/log
  cp $nnetdir/log/* $nnetdir/.backup/log
fi

python nnet/lib/train.py $cmdopts --config $config $train $train_spklist $nnet_dir

exit 0
