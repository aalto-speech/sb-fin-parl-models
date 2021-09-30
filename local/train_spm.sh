#!/bin/bash
# NOTE:
# Requires SentencePiece commandline tools
# See https://github.com/google/sentencepiece
# For this run, install was:
#
# module load cmake
# git clone https://github.com/google/sentencepiece.git 
# cd sentencepiece
# mkdir build
# cd build
# cmake ..
# make -j2

cmd="srun --mem 2G --time 0:30:0"

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <text> <num_units> <outdir>"
  exit 1
fi

num_units="$2"
OUTDIR="$3"
mkdir -p "$OUTDIR"

$cmd sentencepiece/build/src/spm_train --input="$1" \
  --model_prefix="$OUTDIR"/spm."$num_units" \
  --vocab_size="$num_units" \
  --character_coverage=1.0 \
  --model_type="bpe" \
  --unk_id=0 \
  --bos_id=-1 \
  --eos_id=-1 \
  --pad_id=-1
