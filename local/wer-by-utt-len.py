#!/usr/bin/env python3
# This script computes WER correlation with utterance length and
# computes WER in different length quantiles.

import statistics
import bisect
import scipy.stats
from speechbrain.utils import edit_distance
from speechbrain.dataio.wer import print_wer_summary


def text_to_dict(path):
    output = {}
    with open(path) as fi:
        for line in fi:
            uttid, *text = line.strip().split()
            output[uttid] = text
    return output

def split_to_quantiles(textdict, n):
    quantile_cutoffs = statistics.quantiles((len(v) for v in textdict.values()), n=n)
    quantiles = [{} for _ in range(n)]
    for uttid, text in textdict.items():
        v = len(text)
        q_index = bisect.bisect_left(quantile_cutoffs, v)
        quantiles[q_index][uttid] = text
    quantile_strings = [f"l <= {quantile_cutoffs[0]}"] + \
            [f"{quantile_cutoffs[i-1]} < l <= {quantile_cutoffs[i]}" for i in range(1, n-1)] + \
            [f"{quantile_cutoffs[n-2]} < l"]
    return quantiles, quantile_strings

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser("WER by utt length")
    parser.add_argument("ref", help="Path to reference text")
    parser.add_argument("--text", 
            help="Path to hypothesis text, can be specified any number of times",
            action="append", default=[])
    parser.add_argument("--num-quantiles", default=4)
    args = parser.parse_args()
    ref_dict = text_to_dict(args.ref)
    ref_quantiles, cutoffs = split_to_quantiles(ref_dict, n=args.num_quantiles)
    print("Reference length quantile cutoffs:")
    print("\t" + "\n\t".join(cutoffs))
    hyps = {}
    for hyp in args.text:
        print()
        hyp_dict = text_to_dict(hyp)
        hyp_details = edit_distance.wer_details_by_utterance(ref_dict, hyp_dict)
        hyp_summary = edit_distance.wer_summary(hyp_details)
        print(hyp, ": overall")
        print_wer_summary(hyp_summary)
        for ref_quantile, name in zip(ref_quantiles, cutoffs):
            quantile_details = edit_distance.wer_details_by_utterance(ref_quantile, hyp_dict)
            quantile_summary = edit_distance.wer_summary(quantile_details)
            print(hyp, ":", name)
            print_wer_summary(quantile_summary)
        len_wer_corr = scipy.stats.pearsonr(
                [det["num_ref_tokens"] for det in hyp_details],
                [det["WER"] for det in hyp_details]
        )
        print("Ref length to WER correlation [Pearson R, p-value]:", len_wer_corr)
