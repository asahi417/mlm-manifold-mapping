"""Rewrite sentence based on manifold mapping onto the low-perplexity space."""
import argparse
import json
import os
import logging
from pprint import pprint
from m3 import Rewriter

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def main():
    parser = argparse.ArgumentParser(
        description='Rewrite sentence based on manifold mapping onto the low-perplexity space.'
    )
    # config
    parser.add_argument('-m', '--model', help='transformer LM', default='albert-base-v2', type=str)
    parser.add_argument('-n', '--max-n-iteration', help='', default=5, type=int)
    parser.add_argument('-k', '--topk', help='', default=5, type=int)
    parser.add_argument('-l', '--length', help='max sequence length of language model', default=64, type=int)
    parser.add_argument('-b', '--batch-size', help='Batch size', default=512, type=int)
    parser.add_argument('-o', '--output-dir', help='Directory to output', default='./prompts/analogy', type=str)
    # input
    parser.add_argument('-f', '--file-path', help='path to file', default=None, type=str)
    parser.add_argument('-i', '--input-sentence', help='path to file', default=None, type=str, nargs='+')
    parser.add_argument('-e', '--export-file-path', help='path to file', default='output.json', type=str)
    opt = parser.parse_args()

    rewriter = Rewriter(opt.model, max_length=opt.length)

    if opt.file_path is not None:
        assert os.path.exists(opt.file_path), f'file not found at {opt.file_path}'
        with open(opt.file_path) as f:
            input_sentences = [i for i in f.read().split('\n') if len(i) > 0]
    else:
        assert opt.file_path is not None or opt.input_sentence is not None, \
            "either of `--file-path` or `--input-sentence` should be specified."
        input_sentences = opt.input_sentence
    output = rewriter.generate(
        input_sentences,
        max_n_iteration=opt.max_n_iteration,
        topk=opt.topk,
        batch_size=opt.batch_size
    )
    pprint(output)
    if os.path.dirname(opt.export_file_path) != '':
        os.makedirs(os.path.dirname(opt.export_file_path), exist_ok=True)
    with open(opt.export_file_path, 'w') as f:
        json.dump({i: o for i, o in zip(input_sentences, output)}, f)


if __name__ == '__main__':
    main()
