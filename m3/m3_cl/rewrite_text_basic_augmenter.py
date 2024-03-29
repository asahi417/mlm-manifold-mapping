""" Text augmentation by simple baselines. """
import argparse
import json
import os
import logging
from datasets import load_dataset
from m3 import text_augmenter

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def main():
    parser = argparse.ArgumentParser(description='Text augmentation by simple baselines.')
    # config
    parser.add_argument('-a', '--augment-type', help='', default='word_swapping_random', type=str)
    parser.add_argument('-t', '--transformations-per-example', help='', default=2, type=int)
    parser.add_argument('-m', '--max-token-size', help='', default=256, type=int)
    # input
    parser.add_argument('-f', '--file-path', help='path to file', default=None, type=str)
    parser.add_argument('-i', '--input-sentence', help='input sentence', default=None, type=str, nargs='+')
    parser.add_argument('-d', '--dataset', help='huggingface dataset', default=None, type=str)
    parser.add_argument('-s', '--dataset-split', help='huggingface dataset split', default='train', type=str)
    parser.add_argument('--dataset-name', help='huggingface dataset name', default=None, type=str)
    parser.add_argument('--dataset-column', help='huggingface dataset column', default=None, type=str)
    parser.add_argument('-e', '--export-file-path', help='path to file', default='output.json', type=str)
    opt = parser.parse_args()

    input_sentences = None
    if opt.file_path is not None:
        assert os.path.exists(opt.file_path), f'file not found at {opt.file_path}'
        with open(opt.file_path) as f:
            input_sentences = [i for i in f.read().split('\n') if len(i) > 0]
    elif opt.input_sentence is not None:
        input_sentences = opt.input_sentence
    elif opt.dataset is not None:
        if opt.dataset_name is not None:
            data = load_dataset(opt.dataset, opt.dataset_name, split=opt.dataset_split)
        else:
            data = load_dataset(opt.dataset, split=opt.dataset_split)
        input_sentences = data[opt.dataset_column]
    assert input_sentences is not None, 'input is not specified'

    logging.info(f'total: {len(input_sentences)} sentences')
    input_sentences = [i for i in input_sentences if len(i.split(' ')) < opt.max_token_size]
    logging.info(f'filtered: {len(input_sentences)}')

    output = text_augmenter(
        input_sentences, augment_type=opt.augment_type, transformations_per_example=opt.transformations_per_example
    )
    if os.path.dirname(opt.export_file_path) != '':
        os.makedirs(os.path.dirname(opt.export_file_path), exist_ok=True)
    with open(opt.export_file_path, 'w') as f:
        json.dump({i: [o, list(range(len(o)))] for o, i in zip(output, input_sentences)}, f)


if __name__ == '__main__':
    main()
