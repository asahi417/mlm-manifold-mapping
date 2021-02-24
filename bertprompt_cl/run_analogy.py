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
    parser.add_argument('-k', '--topk', help='Filter to top k token prediction', default=10, type=int)
    parser.add_argument('-o', '--output-dir', help='Directory to output', default='./prompts/analogy', type=str)
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
        prompts = list(chain(*[p[k][0][1:] for p in list_prompt]))
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
    path = '{}/prompt_dict.{}.{}.{}*json'.format(opt.output_dir, opt.data, opt.transformers_model, opt.topk)
    list_prompt = glob(path)
    assert len(list_prompt), path
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
            _, data, model, topk, _ = filename.split('.')
            output_file = '{}/result.{}.{}.{}.best.pkl'.format(
                opt.output_dir, data, model, topk)
        else:
            _, data, model, topk, n_blank, n_blank_b, n_blank_e = filename.split('.')
            output_file = '{}/result.{}.{}.{}.{}.{}.{}.pkl'.format(
                opt.output_dir, data, model, topk, n_blank, n_blank_b, n_blank_e)
        val, test = bertprompt.get_analogy_data(data)
        full_data = val + test

        # get data
        list_answer = [data_['answer'] for data_ in full_data]
        list_choice = [data_['choice'] for data_ in full_data]
        partition = bertprompt.get_partition(list_choice)

        def _main(reverse: bool = False):
            if reverse:
                output_file_ = output_file.replace('.pkl', '.reverse.pkl')
            else:
                output_file_ = output_file
            if os.path.exists(output_file_):
                with open(output_file_, "rb") as fp:
                    score_flat = pickle.load(fp)
                return score_flat

            list_p = []
            for data_ in full_data:
                h, t = data_['stem']
                if reverse:
                    all_template, all_score = prompt_dict['||'.join([t, h])]
                else:
                    all_template, all_score = prompt_dict['||'.join([h, t])]
                template = all_template[-1]
                assert h in template and t in template, '{} and {} not in {}'.format(h, t, template)
                list_p.append([template.replace(h, h_c).replace(t, t_c) for h_c, t_c in data_['choice']])

            prompter = bertprompt.Prompter(model, opt.length)
            score_flat = prompter.get_perplexity(list(chain(*list_p)), batch_size=opt.batch)
            with open(output_file_, 'wb') as fp:
                pickle.dump(score_flat, fp)

            return score_flat

        def _accuracy(__score_flat):
            score = [__score_flat[s_:e_] for s_, e_ in partition]
            accuracy = []
            assert len(score) == len(list_answer)
            for a, s in zip(list_answer, score):
                p = s.index(min(s))
                accuracy.append(int(a == p))
            return sum(accuracy) / len(accuracy)

        _score_flat = _main()
        _score_flat_r = _main(True)
        _score_flat_c = list(map(lambda x: sum(x), zip(_score_flat, _score_flat_r)))
        accuracy_full[filename] = _accuracy(_score_flat)
        accuracy_full[filename + '.reverse'] = _accuracy(_score_flat_r)
        accuracy_full[filename + '.combine'] = _accuracy(_score_flat_c)
    logging.info('All result:\n{}'.format(accuracy_full))
    with open('{}/result.{}.{}.{}.json'.format(opt.output_dir, opt.data, opt.transformers_model, opt.topk), 'w') as f:
        json.dump(accuracy_full, f)
    logging.info('exported to {}/result.json'.format(opt.output_dir))


if __name__ == '__main__':
    main()

