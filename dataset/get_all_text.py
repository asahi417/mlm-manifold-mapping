import os
import json
from os.path import join as pj

target_dataset = ["chemprot", "citation_intent", "rct-sample", "sciie"]  # "hyperpartisan_news"
export_dir = 'all_texts'
os.makedirs(export_dir, exist_ok=True)


def jsonline_loader(_file):
    with open(_file) as f:
        return [json.loads(l) for l in f.read().split('\n') if len(l) > 0]


if __name__ == '__main__':
    for i in target_dataset:
        with open(pj(export_dir, f'{i}.dev.txt'), 'w') as f:
            f.write('\n'.join([k['text'] for k in jsonline_loader(pj(i, "dev.jsonl"))]))
        with open(pj(export_dir, f'{i}.test.txt'), 'w') as f:
            f.write('\n'.join([k['text'] for k in jsonline_loader(pj(i, "test.jsonl"))]))
        with open(pj(export_dir, f'{i}.train.txt'), 'w') as f:
            f.write('\n'.join([k['text'] for k in jsonline_loader(pj(i, "train.jsonl"))]))
