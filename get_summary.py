import json
import os
import requests

import pandas as pd


MODEL = ['roberta-base', 'albert-base-v2']
DATA = ["yelp_review", "chemprot", "citation_intent", "rct_sample", "sciie", "amcd", "tweet_eval_irony",
        "tweet_eval_hate", "tweet_eval_emotion"]
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
            if m == 'albert-base-v2' and d != 'rct-sample':
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
                            f"https://huggingface.co/{ORG}/m3-experiment-{m}-{d}-replace/raw/main/{metric_file}.json",
                            f"{m}-{d}-replace.{metric_file}.json"
                        ),
                        add={"model": m, "data": d, 'version': 'replace'}
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
                        f"https://huggingface.co/{ORG}/m3-experiment-{m}-{d}-add-v3/raw/main/{metric_file}.json",
                        f"{m}-{d}-add-v3.{metric_file}.json"
                    ),
                    add={"model": m, "data": d, 'version': 'add-v3'}
                )
            )
            if d != 'rct-sample':
                output.append(
                    format_metric(
                        download(
                            f"https://huggingface.co/{ORG}/m3-experiment-{m}-{d}-replace-v2/raw/main/{metric_file}.json",
                            f"{m}-{d}-replace-v2.{metric_file}.json"
                        ),
                        add={"model": m, "data": d, 'version': 'replace-v2'}
                    )
                )
    df = pd.DataFrame(output)
    df[['eval_f1', 'eval_f1_macro', 'eval_accuracy']] = df[['eval_f1', 'eval_f1_macro', 'eval_accuracy']] * 100
    tmp = []
    for (m, d), g in df.groupby(by=['model', 'data']):
        entry = {'model': m, 'data': d}
        entry.update({k: v for v, k in zip(
            g[g.version == 'vanilla'][['eval_f1_macro', 'eval_accuracy']].values[0].tolist(),
            ['Macro F1 (vanilla)', 'Accuracy (vanilla)'])})
        entry.update({k: v for v, k in zip(
            g[g.version == 'add-v2'][['eval_f1_macro', 'eval_accuracy']].values[0].tolist(),
            ['Macro F1 (add v2)', 'Accuracy (add v2)'])})
        entry.update({k: v for v, k in zip(
            g[g.version == 'add-v3'][['eval_f1_macro', 'eval_accuracy']].values[0].tolist(),
            ['Macro F1 (add v3)', 'Accuracy (add v3)'])})
        gain_v2 = - g[g.version == 'vanilla'][['eval_f1_macro', 'eval_accuracy']].values \
            + g[g.version == 'add-v2'][['eval_f1_macro', 'eval_accuracy']].values
        entry.update({k: i for i, k in zip(gain_v2[0].tolist(), ['Macro F1 Gain (v2)', 'Accuracy Gain (v2)'])})
        gain_v3 = - g[g.version == 'vanilla'][['eval_f1_macro', 'eval_accuracy']].values \
            + g[g.version == 'add-v3'][['eval_f1_macro', 'eval_accuracy']].values
        entry.update({k: i for i, k in zip(gain_v3[0], ['Macro F1 Gain (v3)', 'Accuracy Gain (v3)'])})
        tmp.append(entry)
    df_gain = pd.DataFrame(tmp)
    return df, df_gain


if __name__ == '__main__':
    full_output, gain = get_result()
    full_output.to_csv(f'{EXPORT_DIR}/summary.csv')
    gain.to_csv(f'{EXPORT_DIR}/summary.gain.csv')

    # full_output, gain = get_result('metric_summary.edit')
    # full_output.to_csv(f'{EXPORT_DIR}/summary.edit.csv')
    # gain.to_csv(f'{EXPORT_DIR}/summary.gain.edit.csv')
