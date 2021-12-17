#!/usr/bin/env python3
import pathlib

def read_texts(filepath):
    with open(filepath) as fin:
        for line in fin:
            uttid, *text = line.strip().split()
            yield uttid, text

def filter_repetitions(seq, max_repetitions):
    seq = list(seq)
    output = []
    for n in range(1, len(seq)):
        iterator = enumerate(seq)
        # Fill first buffers:
        buffers = [[next(iterator)[1]] for _ in range(n)]
        for seq_index, token in iterator:
            current_buffer = seq_index % n
            if token != buffers[current_buffer][-1]:
                # No repeat, we can flush some tokens
                buf_len = sum(map(len, buffers))
                flush_start = (current_buffer-buf_len) % n
                # Keep n-1 tokens, but possibly mark some for removal
                for flush_index in range(buf_len - buf_len%n):
                    if (buf_len - flush_index) > n-1:
                        to_flush = buffers[(flush_index + flush_start) % n].pop(0)
                    else:
                        to_flush = None
                    # Here, repetitions get removed:
                    if (flush_index // n < max_repetitions) and to_flush is not None:
                        output.append(to_flush)
                    elif (flush_index // n >= max_repetitions) and to_flush is None:
                        output.append(to_flush)
            buffers[current_buffer].append(token)
        # At the end, final flush
        current_buffer += 1
        buf_len = sum(map(len, buffers))
        flush_start = (current_buffer-buf_len) % n
        for flush_index in range(buf_len):
            to_flush = buffers[(flush_index + flush_start) % n].pop(0)
            # Here, repetitions just get removed:
            if flush_index // n < max_repetitions:
                output.append(to_flush)
        seq = []
        to_delete = 0
        for token in output:
            if token is None:
                to_delete += 1
            elif to_delete > 0:
                to_delete -= 1
            else:
                seq.append(token)
        output = []
        # Don't need to iterate over impossible n values:
        if len(seq) < (max_repetitions+1)*(n+1):
            break
    return seq

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("textfile", type=pathlib.Path)
    parser.add_argument("--max-repetitions", type=int, default=3)
    parser.add_argument("--save-modified", type=pathlib.Path, default=None)
    args = parser.parse_args()
    modified_uttids = []
    for uttid, text in read_texts(args.textfile):
        filtered = filter_repetitions(text, args.max_repetitions)
        print(uttid, " ".join(filtered))
        if filtered != text:
            modified_uttids.append(uttid)
    if args.save_modified is not None:
        with open(args.save_modified, "w") as fo:
            for uttid in modified_uttids:
                print(uttid, file=fo)
