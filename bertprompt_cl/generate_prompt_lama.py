""" Generate prompt for LAMA """
import argparse
import json
import os
import logging
import shutil
from glob import glob
import bertprompt


def get_options():
    parser = argparse.ArgumentParser(description='Generate prompt for LAMA')
    parser.add_argument('-t', '--transformers-model',
                        help='Language model alias from transformers model hub (multiple models is provided by `,`)',
                        required=True, type=str)
    parser.add_argument('-r', '--revision', help='The number of revision by language model', default=15, type=int)
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
    models = sorted(opt.transformers_model.split(','))
    models_str = '_'.join(models)
    logging.info('GENERATE PROMPT FOR LAMA: {}'.format(models))
    data = bertprompt.get_lama_data(transformers_model=models)
    for i, model in enumerate(models):
        prompter = bertprompt.Prompter(model, opt.length)
        mask = prompter.tokenizer.mask_token
        # get flattened template list
        vocab_to_keep, seed_prompt = [], []
        for data_type, sub_data in data.items():
            for rel_type, subsub_data in sub_data.items():
                for n, subsubsub_data in enumerate(subsub_data):
                    masked_prompt = subsubsub_data['prompt'].replace(subsubsub_data['obj_label'], mask)
                    if masked_prompt not in seed_prompt:
                        vocab_to_keep.append([subsubsub_data['sub_label'], mask])
                        seed_prompt.append(masked_prompt)
        # language model inference
        logging.info('Experiment {}/{}'.format(i + 1, len(models)))
        logging.info('\t * model          : {}'.format(model))
        logging.info('\t * unique template: {}'.format(len(seed_prompt)))
        filename = '{}/{}/prompt_dict.{}.{}.{}.json'.format(opt.output_dir, models_str, model, opt.topk, opt.revision)
        if os.path.exists(filename):
            logging.info('skip as the output found at: {}'.format(filename))
            continue
        output_dict = {}
        total_range = range(0, len(seed_prompt), opt.max_data_size)
        for n_, n in enumerate(total_range):
            end = min(n + opt.max_data_size, len(seed_prompt))
            logging.info('sub-experiment {}/{} ({}:{})'.format(n_, len(total_range), n, end))
            filename_ = filename.replace('.json', '.sub.{}.{}.json'.format(n_, opt.max_data_size))
            if os.path.exists(filename_):
                logging.info('\t * loading cache')
                with open(filename_, 'r') as f:
                    output_dict_tmp = json.load(f)
            else:
                seed_prompt_sub = seed_prompt[n:end]
                vocab_to_keep_sub = vocab_to_keep[n:end]
                output_dict_tmp = prompter.generate(
                    seed_sentences=seed_prompt_sub,
                    vocab_to_keep=vocab_to_keep_sub,
                    batch_size=opt.batch,
                    topk=opt.topk,
                    n_revision=opt.revision)
                with open(filename_, 'w') as f:
                    json.dump(output_dict_tmp, f)
            output_dict.update(output_dict_tmp)

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        logging.info('experiment finished, exporting result to {}'.format(filename))
        with open(filename, 'w') as f:
            json.dump(output_dict, f)
        with open(filename.replace('.json', '.top.json'), 'w') as f:
            json.dump({k: [v[0][-1], v[1][-1]] for k, v in output_dict.items()}, f)
        logging.info('deleting cached files')
        for p in glob('{}/{}/prompt_dict.*.sub.*.json'.format(opt.output_dir, models_str)):
            shutil.rmtree(p)


if __name__ == '__main__':
    main()
