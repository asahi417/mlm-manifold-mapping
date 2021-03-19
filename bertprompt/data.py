""" Dataset downloader """
import os
import logging
import requests
import zipfile
import json
from typing import Dict, List
import transformers
__all__ = ('get_analogy_data', 'get_lama_data')
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
MASK = '[MASK]'
relations_concept_squad = [{"relation": "test", "template": None}]
default_cache_dir_lama = '{}/.cache/bertprompt/data/lama'.format(os.path.expanduser('~'))
root_url_lama = 'https://dl.fbaipublicfiles.com/LAMA/data.zip'

default_cache_dir_analogy = '{}/.cache/bertprompt/data/analogy'.format(os.path.expanduser('~'))
root_url_analogy = 'https://github.com/asahi417/AnalogyTools/releases/download/0.0.0'


def wget(url, cache_dir):
    logging.debug('downloading zip file from {}'.format(url))
    os.makedirs(cache_dir, exist_ok=True)
    filename = os.path.basename(url)
    with open('{}/{}'.format(cache_dir, filename), "wb") as f:
        r = requests.get(url)
        f.write(r.content)

    with zipfile.ZipFile('{}/{}'.format(cache_dir, filename), 'r') as zip_ref:
        zip_ref.extractall(cache_dir)
    os.remove('{}/{}'.format(cache_dir, filename))


def get_analogy_data(data_name: str, cache_dir: str = default_cache_dir_analogy):
    """ Get SAT-type dataset: a list of (answer: int, prompts: list, stem: list, choice: list)"""
    assert data_name in ['sat', 'u2', 'u4', 'google', 'bats'], 'unknown data: {}'.format(data_name)
    if not os.path.exists('{}/{}'.format(cache_dir, data_name)):
        url = '{}/{}.zip'.format(root_url_analogy, data_name)
        wget(url, cache_dir)

    with open('{}/{}/test.jsonl'.format(cache_dir, data_name), 'r') as f:
        test = list(filter(None, map(lambda x: json.loads(x) if len(x) > 0 else None, f.read().split('\n'))))
    with open('{}/{}/valid.jsonl'.format(cache_dir, data_name), 'r') as f:
        val = list(filter(None, map(lambda x: json.loads(x) if len(x) > 0 else None, f.read().split('\n'))))
    return val, test


def parse_template(template, subject_label):
    return template.replace("[X]", subject_label).replace("[Y]", MASK)


def get_lama_data(cache_dir: str = None,
                  vocab: Dict = None,
                  transformers_model: List = None,
                  drop_duplicated_prompt: bool = False):
    """ Get LAMA dataset.

    Parameters
    ----------
    cache_dir : str
        To change the directory to cache data.
    vocab : dict
        Dictionary where
    transformers_model
    drop_duplicated_prompt : bool
        Data originally has duplicated entries where they share same prompt i.e. same template and subject. This option
        will drop such a duplication by picking up first one.

    Returns
    -------

    """
    cache_dir = default_cache_dir_lama if cache_dir is None else cache_dir
    vocab_list = []
    if transformers_model:
        if type(transformers_model) is str:
            transformers_model = [transformers_model]
        try:
            vocab_list = [transformers.AutoTokenizer.from_pretrained(m).vocab for m in transformers_model]
        except ValueError:
            vocab_list = [transformers.AutoTokenizer.from_pretrained(m, local_files_only=True).vocab for m in transformers_model]
    if vocab:
        vocab_list += [vocab]

    if not os.path.exists(cache_dir):
        wget(root_url_lama, cache_dir)

    full_set = {}

    def load_jsonl(__file):
        with open(__file, 'r') as _f:
            return list(filter(None, map(lambda x: json.loads(x) if len(x) else None, _f.read().split('\n'))))

    def get_value(_dict, template: str = None, is_squad=False):
        try:
            # Squad does not have subject label
            if is_squad:
                _dict['sub_label'] = ''
            # single character object could be a broken entry
            if len(_dict['obj_label']) == 1:
                return None
            if vocab_list:  # make sure obj_label is in vocabulary
                assert vocab_list[0][_dict['obj_label']]
                assert all(v[_dict['obj_label']] for v in vocab_list)
            if template:
                _dict['prompt'] = parse_template(template, _dict['sub_label'])
            else:
                assert len(_dict['masked_sentences']) == 1 and type(_dict['masked_sentences']) is list
                # _dict['prompt'] = _dict['masked_sentences'][0].replace('[MASK]', _dict['obj_label'])
                _dict['prompt'] = _dict['masked_sentences'][0]
            return {k: _dict[k] for k in ['obj_label', 'sub_label', 'prompt']}
        except KeyError:
            return None

    logging.debug('processing data')

    for i in ['ConceptNet', 'Google_RE', 'Squad', 'TREx']:
        if i == 'TREx':
            relation = load_jsonl('{}/data/relations.jsonl'.format(cache_dir))
        elif i == 'Google_RE':
            relation = relations_google
        else:
            relation = relations_concept_squad

        full_set[i] = {}
        for r in relation:
            if i == 'Google_RE':
                _file = '{}/data/{}/{}_test.jsonl'.format(cache_dir, i, r['relation'])
            else:
                _file = '{}/data/{}/{}.jsonl'.format(cache_dir, i, r['relation'])

            if not os.path.exists(_file):
                logging.debug('\t FILE SKIPPED: file not found {}'.format(_file))
            else:
                data = list(filter(None, map(
                    lambda x: get_value(x, template=r['template'], is_squad=i == 'Squad'), load_jsonl(_file))))
                if drop_duplicated_prompt:
                    # pick one entry from what share same prompt i.e. same template and subject
                    unique_prompt = list(set([d['prompt'] for d in data]))
                    data = [list(filter(lambda x: x['prompt'] == p, data))[0] for p in unique_prompt]
                full_set[i][r['relation']] = data
                logging.debug('\t * {}/{}: {}'.format(i, r['relation'], len(data)))
        logging.debug('\t * {}: {}'.format(i, sum(len(i) for i in full_set[i].values())))
    return full_set

