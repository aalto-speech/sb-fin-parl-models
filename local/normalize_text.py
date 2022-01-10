#!/usr/bin/env python3
import sys
from preprocess_lm_data import normalize

for line in sys.stdin:
    uttid, *tokens = line.strip().split()
    text = " ".join(tokens)
    print(uttid, normalize(text))
    
