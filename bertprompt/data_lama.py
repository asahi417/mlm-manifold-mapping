import os
import logging
import requests
import zipfile
import json
from typing import Dict, List
import transformers

url = 'https://dl.fbaipublicfiles.com/LAMA/data.zip'
relations_google = [
    {
        "relation": "place_of_birth",
        "template": "[X] was born in [Y] .",
        "template_negated": "[X] was not born in [Y] .",
    },
    {
        "relation": "date_of_birth",
        "template": "[X] (born [Y]).",
        "template_negated": "[X] (not born [Y]).",
    },
    {
        "relation": "place_of_death",
        "template": "[X] died in [Y] .",
        "template_negated": "[X] did not die in [Y] .",
    },
]
relations_concept_squad = [{"relation": "test", "template": None}]
default_cache_dir = '{}/.cache/bertprompt/data'.format(os.path.expanduser('~'))


def parse_template(template, subject_label, object_label):
    return template.replace("[X]", subject_label).replace("[Y]", object_label)


def get_lama_data(cache_dir: str = default_cache_dir, vocab: Dict = None, transformers_model: List = None):
    vocab_list = []
    if transformers_model:
        if type(transformers_model) is str:
            transformers_model = [transformers_model]
        vocab_list = [transformers.AutoTokenizer.from_pretrained(m).vocab for m in transformers_model]
    if vocab:
        vocab_list += [vocab]

    if not os.path.exists(cache_dir):
        logging.debug('downloading zip file from {}'.format(url))
        os.makedirs(cache_dir, exist_ok=True)
        filename = os.path.basename(url)
        with open('{}/{}'.format(cache_dir, filename), "wb") as f:
            r = requests.get(url)
            f.write(r.content)

        with zipfile.ZipFile('{}/{}'.format(cache_dir, filename), 'r') as zip_ref:
            zip_ref.extractall(os.path.dirname(cache_dir))
        os.remove('{}/{}'.format(cache_dir, filename))

    full_set = {}

    def load_jsonl(__file):
        with open(__file, 'r') as _f:
            return list(filter(None, map(lambda x: json.loads(x) if len(x) else None, _f.read().split('\n'))))

    def get_value(_dict, template: str = None):
        try:
            if vocab_list:  # make sure obj_label is in vocabulary
                assert vocab_list[0][_dict['obj_label']]
                assert all(v[_dict['obj_label']] for v in vocab_list)
            if template:
                _dict['prompt'] = parse_template(template, _dict['sub_label'], _dict['obj_label'])
            else:
                assert len(_dict['masked_sentences']) == 1 and type(_dict['masked_sentences']) is list
                _dict['prompt'] = _dict['masked_sentences'][0].replace('[MASK]', _dict['obj_label'])
            return {k: _dict[k] for k in ['obj_label', 'sub_label', 'prompt']}
        except KeyError:
            return None

    logging.debug('processing data')

    for i in ['ConceptNet', 'Google_RE', 'Squad', 'TREx']:
        if i == 'TREx':
            relation = load_jsonl('{}/relations.jsonl'.format(cache_dir))
        elif i == 'Google_RE':
            relation = relations_google
        else:
            relation = relations_concept_squad

        full_set[i] = {}
        for r in relation:
            if i == 'Google_RE':
                _file = '{}/{}/{}_test.jsonl'.format(cache_dir, i, r['relation'])
            else:
                _file = '{}/{}/{}.jsonl'.format(cache_dir, i, r['relation'])

            if not os.path.exists(_file):
                logging.debug('\t FILE SKIPPED: file not found {}'.format(_file))
            else:
                data = list(filter(None, map(lambda x: get_value(x, template=r['template']), load_jsonl(_file))))
                full_set[i][r['relation']] = data
                logging.debug('\t * {}/{}: {}'.format(i, r['relation'], len(data)))
        logging.debug('\t * {}: {}'.format(i, sum(len(i) for i in full_set[i].values())))
    return full_set

