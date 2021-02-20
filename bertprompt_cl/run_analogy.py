# import argparse
# import logging
# import json
# import pickle
# import os
# from itertools import chain, product
# from glob import glob
# import bertprompt
#
#
# def get_options():
#     parser = argparse.ArgumentParser(description='Run analogy test')
#     parser.add_argument('-t', '--transformers-model', help='language model alias from transformers model hub',
#                         required=True, type=str)
#     parser.add_argument('-r', '--revision', help='The number of revision by language model', default=15, type=int)
#     parser.add_argument('-d', '--data', help='Data name: sat/u2/u4/google/bats', default='bats', type=str)
#     parser.add_argument('-l', '--length', help='Max length of language model', default=32, type=int)
#     parser.add_argument('-b', '--batch', help='Batch size', default=512, type=int)
#     parser.add_argument('-o', '--output-dir', help='Directory to output', default='./prompts/analogy', type=str)
#     parser.add_argument('--debug', help='Show debug log', action='store_true')
#     return parser.parse_args()
#
#
# def get_partition(_list):
#     length = list(map(lambda x: len(x), _list))
#     return list(map(lambda x: [sum(length[:x]), sum(length[:x + 1])], range(len(length))))
#
#
# def main():
#     opt = get_options()
#     level = logging.DEBUG if opt.debug else logging.INFO
#     logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=level, datefmt='%Y-%m-%d %H:%M:%S')
#     val, test = bertprompt.get_analogy_data(opt.data)
#     full_data = val + test
#     for _file in glob('{}/prompt_dict.{}.{}.*json'.format(opt.output_dir, opt.data, opt.transformers_model)):
#         with open(_file, 'r') as f:
#             prompt_dict = json.load(f)
#
#     output_file = '{}/analogy_result.{}.{}.{}.{}.pkl'.format(opt.output_dir, opt.data, opt.transformers_model, opt.n_blank)
#     # if False:
#     if os.path.exists(output_file):
#         with open(output_file, "rb") as fp:
#             score = pickle.load(fp)
#         list_answer = [data['answer'] for data in full_data]
#     else:
#         dict_file = '{}/{}.{}.{}.{}.json'.format(export_dit, dataset, model, n_blank, seed_type)
#         if not os.path.exists(dict_file):
#             return
#
#         with open(dict_file, 'r') as f:
#             prompt_dict = json.load(f)
#         list_answer = []
#         list_prompt = []
#         for data in full_data:
#             list_answer.append(data['answer'])
#             h, t = data['stem']
#             all_template, all_score = prompt_dict['||'.join([h, t])]
#             template = all_template[-1]
#             assert h in template and t in template, '{} and {} not in {}'.format(h, t, template)
#             list_prompt.append([template.replace(h, h_c).replace(t, t_c) for h_c, t_c in data['choice']])
#         partition = get_partition(list_prompt)
#         score = lm.get_perplexity(list(chain(*list_prompt)), batch_size=512)
#         score = [score[s:e] for s, e in partition]
#         with open(output_file, 'wb') as fp:
#             pickle.dump(score, fp)
#     accuracy = []
#     assert len(score) == len(list_answer)
#     for a, s in zip(list_answer, score):
#         p = s.index(min(s))
#         accuracy.append(int(a == p))
#     accuracy = sum(accuracy)/len(accuracy)
#     return accuracy
#
#
# if __name__ == '__main__':
#     for _s, b in product(seed_types, n_blanks):
#         acc = main(b, _s)
#         print('\nseed: {}, blank: {}'.format(_s, b))
#         print(acc)
#
