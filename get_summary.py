import json
import os
import requests
from statistics import mean
import pandas as pd


MODEL = ['roberta-base', 'albert-base-v2']
# DATA = ["chemprot", "citation_intent", "rct_sample", "sciie", "amcd",
#         "tweet_eval_irony", "tweet_eval_hate", "tweet_eval_emotion"]
DATA = ["citation_intent", "rct_sample", "tweet_eval_irony", "tweet_eval_hate", "tweet_eval_emotion"]  # "amcd"
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

            output.append(
                format_metric(
                    download(
                        f"https://huggingface.co/{ORG}/m3-experiment-{m}-{d}-add-v3/raw/main/{metric_file}.json",
                        f"{m}-{d}-add-v3.{metric_file}.json"
                    ),
                    add={"model": m, "data": d, 'version': 'add-v4'}
                )
            )

            output.append(
                format_metric(
                    download(
                        f"https://huggingface.co/{ORG}/m3-experiment-{m}-{d}-add-v3-greedy/raw/main/{metric_file}.json",
                        f"{m}-{d}-add-v3-greedy.{metric_file}.json"),
                    add={"model": m, "data": d, 'version': 'add-v3-greedy'}
                )
            )

            output.append(
                format_metric(
                    download(
                        f"https://huggingface.co/{ORG}/m3-experiment-{m}-{d}-back-translation/raw/main/{metric_file}.json",
                        f"{m}-{d}-back-translation.{metric_file}.json"),
                    add={"model": m, "data": d, 'version': 'back-translation'}
                )
            )
            for augmenter in ["eda", "word_swapping_embedding", "word_swapping_random", "word_swapping_synonym"]:
                augmenter_output = []
                for seed in ["0", "1", "2", "3", "4"]:
                    augmenter_output.append(
                        format_metric(
                            download(
                                f"https://huggingface.co/{ORG}/m3-experiment-{m}-{d}-{augmenter.replace('_', '-')}-{seed}/raw/main/{metric_file}.json",
                                f"{m}-{d}-{augmenter.replace('_', '-')}-{seed}.{metric_file}.json"),
                            add={"model": m, "data": d, 'version': augmenter}
                        )
                    )
                # print(augmenter_output)
                metrics = [os.path.basename(i) for i in METRICS]
                aggregated_mean = {m: mean([i[m] for i in augmenter_output]) for m in metrics}
                aggregated_mean.update({"model": m, "data": d, 'version': f"{augmenter}"})
                output.append(aggregated_mean)

                aggregated_max = {m: max([i[m] for i in augmenter_output]) for m in metrics}
                aggregated_max.update({"model": m, "data": d, 'version': f"{augmenter}/max"})
                output.append(aggregated_max)

                aggregated_min = {m: min([i[m] for i in augmenter_output]) for m in metrics}
                aggregated_min.update({"model": m, "data": d, 'version': f"{augmenter}/min"})
                output.append(aggregated_min)

    df = pd.DataFrame(output)
    df[['eval_f1', 'eval_f1_macro', 'eval_accuracy']] = df[['eval_f1', 'eval_f1_macro', 'eval_accuracy']] * 100

    tmp = []
    for (m, d), g in df.groupby(by=['model', 'data']):
        # entry = {'column', 'model': m, 'data': d}
        entry = {'column': f"{m}/{d}"}
        for v in ['vanilla', 'add-v2', 'add-v3', 'add-v4', 'add-v3-greedy', 'back-translation', "eda", "word_swapping_embedding",
                  "word_swapping_random", "word_swapping_synonym"]:
            entry.update({k: v for v, k in zip(
                g[g.version == v][['eval_f1_macro', 'eval_f1', 'eval_accuracy']].values[0].tolist(),
                [f'Macro F1 ({v})', f'Micro F1 ({v})',  f'Accuracy ({v})'])})
        for v in ['add-v2', 'add-v3', 'add-v4', 'add-v3-greedy', 'back-translation', "eda", "word_swapping_embedding",
                  "word_swapping_random", "word_swapping_synonym"]:
            _gain = - g[g.version == 'vanilla'][['eval_f1_macro', 'eval_f1', 'eval_accuracy']].values \
                   + g[g.version == v][['eval_f1_macro', 'eval_f1', 'eval_accuracy']].values
            entry.update({k: i for i, k in zip(_gain[0].tolist(), [f'Macro F1 Gain ({v})', f'Micro F1 Gain ({v})', f'Accuracy Gain ({v})'])})
        tmp.append(entry)

    df_gain = pd.DataFrame(tmp)
    df_gain.index = df_gain.pop("column")
    df_gain = df_gain.T
    return df, df_gain


if __name__ == '__main__':
    full_output, gain = get_result()
    full_output.to_csv(f'{EXPORT_DIR}/summary.csv')
    gain.to_csv(f'{EXPORT_DIR}/summary.gain.csv')
