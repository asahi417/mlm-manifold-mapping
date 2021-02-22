import argparse
import logging
import json
import pickle
import os
from itertools import chain
from glob import glob
import bertprompt


def get_options():
    parser = argparse.ArgumentParser(description='Run analogy test')
    parser.add_argument('-l', '--length', help='Max length of language model', default=16, type=int)
    parser.add_argument('-b', '--batch', help='Batch size', default=512, type=int)
    parser.add_argument('-o', '--output-dir', help='Directory to output', default='./prompts/analogy', type=str)
    parser.add_argument('--debug', help='Show debug log', action='store_true')
    return parser.parse_args()


def main():
    opt = get_options()
    level = logging.DEBUG if opt.debug else logging.INFO
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=level, datefmt='%Y-%m-%d %H:%M:%S')
    logging.info('RUN ANALOGY TEST WITH PROMPT')
    accuracy_full = {}
    for _file in glob('{}/prompt_dict*json'.format(opt.output_dir)):
        logging.info('Running inference on {}'.format(_file))
        filename = os.path.basename(_file).replace('.json', '')
        _, data, model, n_blank, n_blank_b, n_blank_e = filename.split('.')
        val, test = bertprompt.get_analogy_data(data)
        full_data = val + test
        with open(_file, 'r') as f:
            prompt_dict = json.load(f)
        output_file = '{}/result.{}.{}.{}.{}.{}.pkl'.format(opt.output_dir, data, model, n_blank, n_blank_b, n_blank_e)

        if os.path.exists(output_file):
            with open(output_file, "rb") as fp:
                score = pickle.load(fp)
            list_answer = [data['answer'] for data in full_data]
        else:
            prompter = bertprompt.Prompter(model, opt.length)
            list_answer, list_prompt = [], []
            for data in full_data:
                list_answer.append(data['answer'])
                h, t = data['stem']
                all_template, all_score = prompt_dict['||'.join([h, t])]
                template = all_template[-1]
                assert h in template and t in template, '{} and {} not in {}'.format(h, t, template)
                list_prompt.append([template.replace(h, h_c).replace(t, t_c) for h_c, t_c in data['choice']])
            partition = bertprompt.get_partition(list_prompt)
            score = prompter.get_perplexity(list(chain(*list_prompt)), batch_size=opt.batch)
            score = [score[s:e] for s, e in partition]
            with open(output_file, 'wb') as fp:
                pickle.dump(score, fp)
        accuracy = []
        assert len(score) == len(list_answer)
        for a, s in zip(list_answer, score):
            p = s.index(min(s))
            accuracy.append(int(a == p))
        accuracy = sum(accuracy) / len(accuracy)
        accuracy_full[filename] = accuracy
        logging.info('Accuracy: {}'.format(accuracy))
    logging.info('All result:\n{}'.format(accuracy_full))
    with open('{}/result.json'.format(opt.output_dir), 'w') as f:
        json.dump(accuracy_full, f)
    logging.info('exported to {}/result.json'.format(opt.output_dir))


if __name__ == '__main__':
    main()

