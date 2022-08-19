import json
import os
import requests

import pandas as pd


MODEL = ['albert-base-v2']
# DATA = ["chemprot", "citation_intent", "sciie", "amcd", "tweet_eval_irony", "tweet_eval_hate", "tweet_eval_emotion"]
DATA = ["chemprot", "citation_intent", "sciie", "amcd", "tweet_eval_irony", "tweet_eval_hate", "tweet_eval_emotion"]
TMP_DIR = 'metric_files'
EXPORT_DIR = 'output'
ORG = 'm3'
METRICS = ["test/eval_loss", "test/eval_f1", "test/eval_f1_macro", "test/eval_accuracy"]
os.makedirs(EXPORT_DIR, exist_ok=True)


def download(url, filename):

    print(f'download {url}')
    try:
        with open(f'{TMP_DIR}/{filename}') as f_reader:
            return json.load(f_reader)
    except Exception:
        os.makedirs(TMP_DIR, exist_ok=True)
        with open(f'{TMP_DIR}/{filename}', "wb") as f_reader:
            r = requests.get(url)
            f_reader.write(r.content)
        with open(f'{TMP_DIR}/{filename}') as f_reader:
            return json.load(f_reader)


def format_metric(_dict, add):
    tmp = {os.path.basename(m): _dict[m] for m in METRICS}
    tmp.update(add)
    return tmp


def get_result(metric_file='metric_summary'):
    output = []
    for m in MODEL:
        for d in DATA:
            d = d.replace('_', '-')
            output.append(
                format_metric(
                    download(
                        f"https://huggingface.co/{ORG}/m3-experiment-{m}-{d}-vanilla/raw/main/{metric_file}.json",
                        f"{m}-{d}.{metric_file}.json"),
                    add={"model": m, "data": d, 'version': 'vanilla'}
                )
            )
            output.append(
                format_metric(
                    download(
                        f"https://huggingface.co/{ORG}/m3-experiment-{m}-{d}-add/raw/main/{metric_file}.json",
                        f"{m}-{d}-add.{metric_file}.json"
                    ),
                    add={"model": m, "data": d, 'version': 'add'}
                )
            )
            output.append(
                format_metric(
                    download(
                        f"https://huggingface.co/{ORG}/m3-experiment-{m}-{d}-add-v2/raw/main/{metric_file}.json",
                        f"{m}-{d}-add-v2.{metric_file}.json"
                    ),
                    add={"model": m, "data": d, 'version': 'add-v2'}
                )
            )
            output.append(
                format_metric(
                    download(
                        f"https://huggingface.co/{ORG}/m3-experiment-{m}-{d}-replace/raw/main/{metric_file}.json",
                        f"{m}-{d}-replace.{metric_file}.json"
                    ),
                    add={"model": m, "data": d, 'version': 'replace'}
                )
            )
            output.append(
                format_metric(
                    download(
                        f"https://huggingface.co/{ORG}/m3-experiment-{m}-{d}-replace-v2/raw/main/{metric_file}.json",
                        f"{m}-{d}-replace-v2.{metric_file}.json"
                    ),
                    add={"model": m, "data": d, 'version': 'replace-v2'}
                )
            )
            # output.append(metrics)
    return pd.DataFrame(output)


if __name__ == '__main__':
    full_output = get_result()
    full_output.to_csv(f'{EXPORT_DIR}/summary.csv')
    full_output = get_result('metric_summary.edit')
    full_output.to_csv(f'{EXPORT_DIR}/summary.edit.csv')
