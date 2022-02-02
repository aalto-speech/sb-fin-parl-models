#!/bin/bash
# NOTE:
# Requires SentencePiece commandline tools

cmd="srun --mem 2G --time 0:30:0"
unk=0
bos=-1
eos=-1
pad=-1

. local/parse_options.sh

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <text> <num_units> <outdir>"
  exit 1
fi

num_units="$2"
OUTDIR="$3"
mkdir -p "$OUTDIR"

module load sentencepiece

$cmd spm_train --input="$1" \
  --model_prefix="$OUTDIR"/spm."$num_units" \
  --vocab_size="$num_units" \
  --character_coverage=1.0 \
  --model_type="bpe" \
  --unk_id=$unk \
  --bos_id=$bos \
  --eos_id=$eos \
  --pad_id=$pad
