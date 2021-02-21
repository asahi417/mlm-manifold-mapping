""" Generate prompt for SAT type analogy dataset """
import argparse
import json
import os
from itertools import chain, product
import logging
import bertprompt


def get_options():
    parser = argparse.ArgumentParser(description='Generate prompt for SAT type analogy dataset')
    parser.add_argument('-t', '--transformers-model', help='language model alias from transformers model hub',
                        required=True, type=str)
    parser.add_argument('--n-blank', help='The number of intermediate blank', default='2,3,4', type=str)
    parser.add_argument('--n-blank-b', help='The number of beginning blank', default='0,1,2', type=str)
    parser.add_argument('--n-blank-e', help='The number of last blank', default='0,1,2', type=str)
    parser.add_argument('-d', '--data', help='Data name: sat/u2/u4/google/bats', default='bats', type=str)
    parser.add_argument('-r', '--revision', help='The number of revision by language model', default=100, type=int)
    parser.add_argument('-l', '--length', help='Max length of language model', default=32, type=int)
    parser.add_argument('-b', '--batch', help='Batch size', default=512, type=int)
    parser.add_argument('-k', '--topk', help='Filter to top k token prediction', default=15, type=int)
    parser.add_argument('-o', '--output-dir', help='Directory to output', default='./prompts/analogy', type=str)
    parser.add_argument('--max-data-size', help='Max data size in single run', default=2000, type=int)
    # parser.add_argument('--reverse', help='Get reversed pair', action='store_true')
    parser.add_argument('--debug', help='Show debug log', action='store_true')
    return parser.parse_args()


def main():
    opt = get_options()
    level = logging.DEBUG if opt.debug else logging.INFO
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=level, datefmt='%Y-%m-%d %H:%M:%S')

    prompter = bertprompt.Prompter(opt.transformers_model, opt.length)
    n_blank_list = [int(i) for i in opt.n_blank.split(',')]
    n_blank_b_list = [int(i) for i in opt.n_blank_b.split(',')]
    n_blank_e_list = [int(i) for i in opt.n_blank_e.split(',')]
    val, test = bertprompt.get_analogy_data(opt.data)
    word_pairs = list(chain(*[[i['stem']] + i['choice'] for i in val]))
    word_pairs += list(chain(*[[i['stem']] + i['choice'] for i in test]))
    # if opt.reverse:
    word_pairs += [[p[1], p[0]] for p in word_pairs]
    all_config = list(product(n_blank_list, n_blank_b_list, n_blank_e_list))

    logging.info('GENERATE PROMPT FOR ANALOGY')
    logging.info('\t * data     : {} ({} pairs)'.format(opt.data, len(word_pairs)))
    logging.info('\t * model    : {}'.format(opt.transformers_model))
    logging.info('\t * blank    : {}'.format(n_blank_list))
    logging.info('\t * blank (b): {}'.format(n_blank_b_list))
    logging.info('\t * blank (e): {}'.format(n_blank_e_list))

    for i, (n_blank, n_blank_b, n_blank_e) in enumerate(all_config):
        logging.info('CONFIG {}/{}: blank: {}, blank_b: {}, blank_e: {}'.format(
            i + 1, len(all_config), n_blank, n_blank_b, n_blank_e))
        filename = '{}/prompt_dict.{}.{}.{}.{}.{}.json'.format(
            opt.output_dir, opt.data, opt.transformers_model, n_blank, n_blank_b, n_blank_e)
        if os.path.exists(filename):
            logging.info('skip as the output found at: {}'.format(filename))
            continue

        output_dict = {}
        for n in range(0, len(word_pairs), opt.max_data_size):
            logging.info('subset: {}:{}'.format(n, min(n+opt.max_data_size, len(word_pairs))))
            word_pairs_sub = word_pairs[n:min(n+opt.max_data_size, len(word_pairs))]
            output_dict_tmp = prompter.generate(
                word_pairs_sub,
                n_blank=n_blank,
                n_blank_b=n_blank_b,
                n_blank_e=n_blank_e,
                batch_size=opt.batch,
                topk=opt.topk,
                n_revision=opt.revision)
            output_dict.update(output_dict_tmp)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        logging.info('exporting output to {}'.format(filename))
        with open(filename, 'w') as f:
            json.dump(output_dict, f)


if __name__ == '__main__':
    main()
