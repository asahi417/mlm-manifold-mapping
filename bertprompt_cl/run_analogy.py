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
    parser.add_argument('-t', '--transformers-model',
                        help='Language model alias from transformers model hub (single model only)',
                        required=True, type=str)
    parser.add_argument('-l', '--length', help='Max length of language model', default=16, type=int)
    parser.add_argument('-b', '--batch', help='Batch size', default=512, type=int)
    parser.add_argument('-d', '--data', help='Data name: sat/u2/u4/google/bats', default='bats', type=str)
    parser.add_argument('-k', '--topk', help='Filter to top k token prediction', default=15, type=int)
    parser.add_argument('-o', '--output-dir', help='Directory to output', default='./prompts/analogy', type=str)
    parser.add_argument('--reverse', help='Use the reverse mode', action='store_true')
    parser.add_argument('--debug', help='Show debug log', action='store_true')
    parser.add_argument('--best', help='Use the prompt that achieves the best perplexity', action='store_true')
    return parser.parse_args()


def get_best_prompt(file_list):

    def safe_load(_file):
        with open(_file, 'r') as f:
            return json.load(f)

    list_prompt = list(map(safe_load, file_list))
    optimal_prompt = {}
    for k in list_prompt[0].keys():
        prompts = list(chain(*[p[k][0] for p in list_prompt]))
        scores = list(chain(*[p[k][1] for p in list_prompt]))
        assert len(prompts) == len(scores), '{} != {}'.format(len(prompts), len(scores))
        best_index = scores.index(min(scores))
        optimal_prompt[k] = [[prompts[best_index]], [scores[best_index]]]
    return optimal_prompt


def main():
    opt = get_options()
    level = logging.DEBUG if opt.debug else logging.INFO
    logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=level, datefmt='%Y-%m-%d %H:%M:%S')
    logging.info('RUN ANALOGY TEST WITH PROMPT')
    accuracy_full = {}
    print('{}/prompt_dict.{}.{}.{}*json'.format(opt.output_dir, opt.data, opt.transformers_model, opt.topk))
    list_prompt = glob('{}/prompt_dict.{}.{}.{}*json'.format(opt.output_dir, opt.data, opt.transformers_model, opt.topk))
    assert len(list_prompt)
    if opt.best:
        file_best_prompt = '{}/prompt_dict.{}.{}.{}.best.json'.format(
            opt.output_dir, opt.data, opt.transformers_model, opt.topk)
        if not os.path.exists(file_best_prompt):
            best_prompt = get_best_prompt(list_prompt)
            with open(file_best_prompt, 'w') as f:
                json.dump(best_prompt, f)
        list_prompt = [file_best_prompt]

    for _file in list_prompt:
        logging.info('Running inference on {}'.format(_file))
        filename = os.path.basename(_file).replace('.json', '')

        with open(_file, 'r') as f:
            prompt_dict = json.load(f)
        if 'best' in filename:
            _, data, model, topk = filename.split('.')
            output_file = '{}/result.{}.{}.{}.best.pkl'.format(
                opt.output_dir, data, model, topk)
        else:
            _, data, model, topk, n_blank, n_blank_b, n_blank_e = filename.split('.')
            output_file = '{}/result.{}.{}.{}.{}.{}.{}.pkl'.format(
                opt.output_dir, data, model, topk, n_blank, n_blank_b, n_blank_e)
        val, test = bertprompt.get_analogy_data(data)
        full_data = val + test
        if opt.reverse:
            output_file = output_file.replace('.pkl', '.reverse.pkl')

        if os.path.exists(output_file):
            with open(output_file, "rb") as fp:
                score = pickle.load(fp)
            list_answer = [data['answer'] for data in full_data]
        else:
            prompter = bertprompt.Prompter(model, opt.length)
            list_answer, list_prompt, list_prompt_reverse = [], [], []
            for data in full_data:
                list_answer.append(data['answer'])
                h, t = data['stem']
                all_template, all_score = prompt_dict['||'.join([h, t])]
                all_template_r, all_score_r = prompt_dict['||'.join([t, h])]
                template = all_template[-1]
                template_r = all_template_r[-1]
                assert h in template and t in template, '{} and {} not in {}'.format(h, t, template)
                assert h in template_r and t in template_r, '{} and {} not in {}'.format(h, t, template_r)
                list_prompt.append([template.replace(h, h_c).replace(t, t_c) for h_c, t_c in data['choice']])
                list_prompt_reverse.append([template_r.replace(h, h_c).replace(t, t_c) for h_c, t_c in data['choice']])

            partition = bertprompt.get_partition(list_prompt)
            score = prompter.get_perplexity(list(chain(*list_prompt)), batch_size=opt.batch)
            score = [score[s:e] for s, e in partition]
            if opt.reverse:
                partition_r = bertprompt.get_partition(list_prompt_reverse)
                score_r = prompter.get_perplexity(list(chain(*list_prompt_reverse)), batch_size=opt.batch)
                score_r = [score_r[s:e] for s, e in partition_r]
                score = list(map(lambda x: sum(x), zip(score, score_r)))

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

