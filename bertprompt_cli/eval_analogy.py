import argparse
import logging
import json
import pickle
import os
import re
from itertools import chain
from glob import glob
import bertprompt


def get_options():
    parser = argparse.ArgumentParser(description='Run analogy test with prompt.')
    parser.add_argument('-t', '--transformers-model',
                        help='Language model alias from transformers model hub',
                        default='roberta-large', type=str)
    parser.add_argument('-l', '--length', help='Max length of language model', default=32, type=int)
    parser.add_argument('-b', '--batch', help='Batch size', default=512, type=int)
    parser.add_argument('-d', '--data', help='Data name: sat/u2/u4/google/bats', default='sat', type=str)
    parser.add_argument('-k', '--topk', help='Filter to top k token prediction', default=15, type=int)
    parser.add_argument('-p', '--prompt-dir', help='Directory prompts stored', default='./prompts/analogy', type=str)
    parser.add_argument('-o', '--output-dir', help='Directory to output', default='./eval/analogy', type=str)
    parser.add_argument('--mode', help='Inference mode (ppl/avg)', default='avg', type=str)
    parser.add_argument('--prompt-mode', help='Prompt mode (stem/all)', default='stem', type=str)
    parser.add_argument('--debug', help='Show debug log', action='store_true')
    return parser.parse_args()


def get_best_prompt(file_list):
    """ Get best prompt in terms of ppl. """

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
    path = '{0}/{1}/prompt_dict.{1}.{2}.{3}*json'.format(
        opt.prompt_dir, opt.data, opt.transformers_model, opt.topk)
    list_prompt = sorted(glob(path))
    assert len(list_prompt), path
    file_best_prompt = '{0}/{1}/prompt_dict.{1}.{2}.{3}.best.json'.format(
        opt.prompt_dir, opt.data, opt.transformers_model, opt.topk)
    if file_best_prompt not in list_prompt:
        best_prompt = get_best_prompt(list_prompt)
        with open(file_best_prompt, 'w') as f:
            json.dump(best_prompt, f)
        list_prompt += [file_best_prompt]

    # list_prompt = [file_best_prompt]

    accuracy_full = {}
    os.makedirs(opt.output_dir, exist_ok=True)

    for _file in list_prompt:
        logging.info('Running inference on {}'.format(_file))
        filename = os.path.basename(_file).replace('.json', '')

        with open(_file, 'r') as f_dict:
            prompt_dict = json.load(f_dict)

        cache_file = '{0}/cache/{1}.{2}.{3}.pkl'.format(opt.output_dir, filename, opt.mode, opt.prompt_mode)
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        val, test = bertprompt.get_analogy_data(opt.data)
        if opt.prompt_mode == 'stem':
            all_pairs = []
            all_template = []
            for data_ in val + test:
                h, t = data_['stem']
                template = prompt_dict['||'.join([h, t])][0][-1]
                all_pairs.append([h, t])
                all_template.append(template)
                h_var = re.findall(h, template, re.IGNORECASE)
                t_var = re.findall(t, template, re.IGNORECASE)
                assert len(h_var) and len(t_var), '`{}` and `{}` not in `{}`'.format(h, t, template)

                for h_c, t_c in data_['choice']:
                    all_pairs.append([h_c, t_c])
                    all_template.append(re.sub(r'|'.join(t_var), t_c, re.sub(r'|'.join(h_var), h_c, template)))
        elif opt.prompt_mode == 'all':
            all_pairs = list(chain(*[[o['stem']] + o['choice'] for o in val + test]))
            all_template = [prompt_dict['||'.join([h, t])][0][-1] for h, t in all_pairs]  # get last prompt
        else:
            raise ValueError('unknown prompt_mode: {}'.format(opt.prompt_mode))

        if opt.mode in ['avg', 'cls']:
            # embedding similarity in between averaged embedding
            if os.path.exists(cache_file):
                with open(cache_file, "rb") as fp:
                    embedding = pickle.load(fp)
            else:
                prompter = bertprompt.Prompter(opt.transformers_model, opt.length)
                embedding = prompter.get_embedding(all_template, batch_size=opt.batch, return_cls=opt.mode == 'cls')
                with open(cache_file, 'wb') as fp:
                    pickle.dump(embedding, fp)

            embedding_dict = {str(k): v for k, v in zip(all_pairs, embedding)}

            def cos_similarity(a_, b_):
                return - sum(list(map(lambda x: (x[0] - x[1]) ** 2, zip(a_, b_)))) ** 0.5
                # inner = sum(list(map(lambda x: x[0] * x[1], zip(a_, b_))))
                # norm_a = sum(list(map(lambda x: x * x, a_))) ** 0.5
                # norm_b = sum(list(map(lambda x: x * x, b_))) ** 0.5
                # return inner / (norm_b * norm_a)

            prediction = []
            for single_data in val + test:
                v_choice = [embedding_dict[str(c)] for c in single_data['choice']]
                v_stem = embedding_dict[str(single_data['stem'])]
                sims = [cos_similarity(v_stem, v) for v in v_choice]
                print(sims)
                pred = sims.index(max(sims))
                prediction.append(pred)
        # elif opt.mode == 'ppl':
        #     # validity score based on perplexity
        #     # (A, B) and (C, D) --> P_{A, B}(C, D) is used to compute prompt.
        #     prompter = bertprompt.Prompter(opt.transformers_model, opt.length)
        #     score_flat = prompter.get_perplexity(list(chain(*all_template)), batch_size=opt.batch)
        #     list_choice = [data_['stem'] for data_ in val + test]
        #     partition = bertprompt.get_partition(list_choice)
        #     score = [score_flat[s_:e_] for s_, e_ in partition]
        #     prediction = [s.index(min(s)) for s in score]
        else:
            raise ValueError('unknown mode: {}'.format(opt.mode))

        accuracy = [int(d['answer'] == p) for p, d in zip(prediction, val + test)]
        accuracy_full[filename.replace('prompt_dict.', '')] = {
            'accuracy_valid': 100 * sum(accuracy[:len(val)]) / len(val),
            'accuracy_test': 100 * sum(accuracy[len(val):len(val) + len(test)]) / len(test),
            'accuracy': 100 * sum(accuracy) / len(accuracy)
        }
        logging.info('accuracy: \n{}'.format(
            json.dumps(accuracy_full[filename.replace('prompt_dict.', '')], indent=4, sort_keys=True)
        ))
    logging.info('All result:\n{}'.format(json.dumps(accuracy_full, indent=4, sort_keys=True)))
    path = '{0}/{1}.{2}.{3}.{4}.json'.format(opt.output_dir, opt.data, opt.transformers_model, opt.topk, opt.mode, opt.prompt_mode)
    os.makedirs(opt.output_dir, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(accuracy_full, f)


if __name__ == '__main__':
    main()

