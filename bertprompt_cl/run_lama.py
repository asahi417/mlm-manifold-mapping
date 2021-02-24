""" Generate prompt for LAMA """
import argparse
import json
import os
import logging
import bertprompt


def get_options():
    parser = argparse.ArgumentParser(description='Generate prompt for LAMA')
    parser.add_argument('-t', '--transformers-model',
                        help='Language model alias from transformers model hub (multiple models is provided by `,`)',
                        required=True, type=str)
    parser.add_argument('-r', '--revision', help='The number of revision by language model', default=15, type=int)
    parser.add_argument('-l', '--length', help='Max length of language model', default=32, type=int)
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
    logging.info('RUN LAMA PREDICTION: {}'.format(models))
    data = bertprompt.get_lama_data(transformers_model=models)
    for i, model in enumerate(models):
        prompter = bertprompt.Prompter(model, opt.length)
        mask = prompter.tokenizer.mask_token
        # aggregate flatten data and index to restore the structure
        data_index, obj_token, sub_token, prompt = [], [], [], []
        for data_type, sub_data in data.items():
            for rel_type, subsub_data in sub_data.items():
                for n, subsubsub_data in enumerate(subsub_data):
                    data_index.append([data_type, rel_type, n])
                    sub_token.append(subsubsub_data['sub_label'])
                    obj_token.append(subsubsub_data['obj_label'])
                    prompt.append(subsubsub_data['prompt'].replace(subsubsub_data['obj_label'], mask))
        # language model inference
        logging.info('Experiment {}'.format(i + 1))
        logging.info('\t * model: {}'.format(model))
        logging.info('\t * data : {}'.format(len(prompt)))

        # filename = '{}/{}/result.{}{}.{}.json'.format(opt.output_dir, models_str, model, opt.topk, opt.revision)
        # if os.path.exists(filename):
        #     logging.info('skip as the output found at: {}'.format(filename))
        #     continue
        # output_dict = {}
        # for n in range(0, len(seed_prompt), opt.max_data_size):
        #     seed_prompt_sub = seed_prompt[n:min(n + opt.max_data_size, len(seed_prompt))]
        #     output_dict_tmp = prompter.generate(
        #         seed_sentences=seed_prompt_sub,
        #         vocab_to_keep=vocab_to_keep,
        #         batch_size=opt.batch,
        #         topk=opt.topk,
        #         n_revision=opt.revision)
        #     output_dict.update(output_dict_tmp)
        #
        # os.makedirs(os.path.dirname(filename), exist_ok=True)
        # logging.info('exporting output to {}'.format(filename))
        # with open(filename, 'w') as f:
        #     json.dump(output_dict, f)
        #

if __name__ == '__main__':
    main()
