""" Generate prompt for LAMA """
import argparse
import os
import logging
import shutil
import pickle
from glob import glob
import bertprompt


def get_options():
    parser = argparse.ArgumentParser(description='Generate prompt for LAMA')
    parser.add_argument('-t', '--transformers-model',
                        help='Language model alias from transformers model hub', required=True, type=str)
    parser.add_argument('-r', '--revision', help='The number of revision by language model', default=100, type=int)
    parser.add_argument('-l', '--length', help='Max length of language model', default=256, type=int)
    parser.add_argument('-b', '--batch', help='Batch size', default=512, type=int)
    parser.add_argument('-k', '--topk', help='Filter to top k token prediction', default=15, type=int)
    parser.add_argument('-o', '--output-dir', help='Directory to output', default='./prompts/lama', type=str)
    parser.add_argument('--max-data-size', help='Max data size in single run', default=2000, type=int)
    parser.add_argument('--debug', help='Show debug log', action='store_true')
    return parser.parse_args()


def main():
    opt = get_options()
    level = logging.DEBUG if opt.debug else logging.INFO
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=level, datefmt='%Y-%m-%d %H:%M:%S')
    prompter = bertprompt.Prompter(opt.transformers_model, opt.length)

    # aggregate data
    data = bertprompt.get_lama_data(transformers_model=opt.transformers_model)
    mask = prompter.tokenizer.mask_token
    # get flattened template list
    vocab_to_keep, seed_prompt = [], []
    for data_type, sub_data in data.items():
        for rel_type, subsub_data in sub_data.items():
            for n, subsubsub_data in enumerate(subsub_data):
                masked_prompt = subsubsub_data['prompt'].replace(bertprompt.data.MASK, mask)
                vtk = [subsubsub_data['sub_label'], mask]
                if masked_prompt in seed_prompt and vtk == vocab_to_keep[seed_prompt.index(masked_prompt)]:
                    continue
                vocab_to_keep.append(vtk)
                seed_prompt.append(masked_prompt)

    # language model inference
    logging.info('GENERATE PROMPT FOR LAMA')
    logging.info('\t * model          : {}'.format(opt.transformers_model))
    logging.info('\t * unique template: {}'.format(len(seed_prompt)))

    filename = '{}/{}/prompt_dict.{}.{}.pkl'.format(opt.output_dir, opt.transformers_model, opt.topk, opt.revision)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if os.path.exists(filename):
        logging.info('skip as the output found at: {}'.format(filename))
    files = []
    total_range = range(0, len(seed_prompt), opt.max_data_size)
    for n_, n in enumerate(total_range):
        end = min(n + opt.max_data_size, len(seed_prompt))
        logging.info('sub-experiment {}/{} ({}:{})'.format(n_, len(total_range), n, end))
        filename_ = filename.replace('.pkl', '.sub.{}.{}.pkl'.format(n_, opt.max_data_size))
        seed_prompt_sub = seed_prompt[n:end]
        vocab_to_keep_sub = vocab_to_keep[n:end]
        if not os.path.exists(filename_):
            output_list_tmp = prompter.generate(
                seed_sentences=seed_prompt_sub,
                vocab_to_keep=vocab_to_keep_sub,
                batch_size=opt.batch,
                topk=opt.topk,
                n_revision=opt.revision)
            assert len(seed_prompt_sub) == len(output_list_tmp) == len(vocab_to_keep_sub),\
                str([len(seed_prompt_sub), len(output_list_tmp), len(vocab_to_keep_sub)])
            output_list_tmp = list(zip(output_list_tmp, seed_prompt_sub, vocab_to_keep_sub))
            with open(filename_, "wb") as fp:
                pickle.dump(output_list_tmp, fp)
        files.append(filename_)

    logging.info('experiment finished, exporting result to {}'.format(filename))
    # combine output
    output_list = []
    for _file in files:
        with open(_file, "rb") as fp:
            output_list += pickle.load(fp)
    with open(filename, "wb") as fp:
        pickle.dump(output_list, fp)
    logging.info('deleting cached files')
    for p in glob('{}/{}/prompt_dict.*.sub.*.pkl'.format(opt.output_dir, opt.transformers_model)):
        shutil.rmtree(p)


if __name__ == '__main__':
    main()
