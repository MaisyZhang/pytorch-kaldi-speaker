#!/bin/bash

normalize=false

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
feat=$3
dir=$4

if $normalize; then
  cmdopt_norm="--normalize"
fi

python nnet/lib/extract.py $cmdopt_norm --config $config $nnet_dir "$feat" "$dir"