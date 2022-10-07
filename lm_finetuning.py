"""
- TrainingArguments
https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments
https://huggingface.co/transformers/v3.1.0/_modules/transformers/trainer_utils.html
https://github.com/huggingface/notebooks/blob/main/examples/text_classification.ipynb
"""
import argparse
import json
import logging
import os
import shutil
import urllib.request
import multiprocessing
from os.path import join as pj

import torch
import numpy as np
from huggingface_hub import create_repo
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from ray import tune

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

PARALLEL = bool(int(os.getenv("PARALLEL", 1)))
RAY_RESULTS = os.getenv("RAY_RESULTS", "ray_results")


def internet_connection(host='http://google.com'):
    try:
        urllib.request.urlopen(host)
        return True
    except:
        return False


def get_metrics():
    metric_accuracy = load_metric("accuracy")
    metric_f1 = load_metric("f1")

    def compute_metric_search(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric_f1.compute(predictions=predictions, references=labels, average='micro')

    def compute_metric_all(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            'f1': metric_f1.compute(predictions=predictions, references=labels, average='micro')['f1'],
            'f1_macro': metric_f1.compute(predictions=predictions, references=labels, average='macro')['f1'],
            'accuracy': metric_accuracy.compute(predictions=predictions, references=labels)['accuracy']
        }
    return compute_metric_search, compute_metric_all


def main():
    parser = argparse.ArgumentParser(description='Fine-tuning language model.')
    parser.add_argument('-m', '--model', help='transformer LM', default='albert-base-v2', type=str)
    parser.add_argument('-d', '--dataset', help='', default='m3/multi_domain_document_classification', type=str)
    parser.add_argument('--dataset-name', help='huggingface dataset name', default='citation_intent', type=str)
    parser.add_argument('-l', '--seq-length', help='', default=128, type=int)
    parser.add_argument('--random-seed', help='', default=42, type=int)
    parser.add_argument('--eval-step', help='', default=50, type=int)
    parser.add_argument('-o', '--output-dir', help='Directory to output', default='m3_ckpt', type=str)
    parser.add_argument('-t', '--n-trials', default=10, type=int)
    parser.add_argument('--push-to-hub', action='store_true')
    parser.add_argument('--use-auth-token', action='store_true')
    parser.add_argument('--hf-organization', default=None, type=str)
    parser.add_argument('-a', '--model-alias', help='', default=None, type=str)
    parser.add_argument('--summary-file', default='metric_summary.json', type=str)
    parser.add_argument('--rewrite-dictionary-dir', default=None, type=str)
    parser.add_argument('--rewrite-dictionary-method', default='largest_diff', type=str)
    parser.add_argument('--rewrite-dictionary-split', default=['train'], nargs='+', type=str)
    parser.add_argument('--add-rewrite-text', action='store_true')
    parser.add_argument('--skip-train', action='store_true')
    parser.add_argument('--skip-eval', action='store_true')
    opt = parser.parse_args()
    assert opt.summary_file.endswith('.json'), f'`--summary-file` should be a json file {opt.summary_file}'
    # setup data
    dataset = load_dataset(opt.dataset, opt.dataset_name)
    if opt.rewrite_dictionary_dir is not None:
        for k in opt.rewrite_dictionary_split:
            assert k in dataset.keys(), f'{k} not in {dataset.keys()}'
            rewrite_file = pj(opt.rewrite_dictionary_dir, f'{k}.json')
            assert os.path.exists(rewrite_file), f'file not found: {rewrite_file}'
            with open(rewrite_file) as f:
                v = json.load(f)
            if opt.rewrite_dictionary_method == 'best':
                v = {_k: [_v[0][-1]] for _k, _v in v.items()}
            elif opt.rewrite_dictionary_method == 'largest_diff':
                v = {_k: [_v[0][np.diff(_v[1]).argmin() + 1] if len(_v[1]) > 1 else _v[0][0]] for _k, _v in v.items()}
            elif opt.rewrite_dictionary_method == 'all':
                assert opt.add_rewrite_text
                v = {_k: _v[0] for _k, _v in v.items()}
            elif opt.rewrite_dictionary_method.isdigit():
                _i = int(opt.rewrite_dictionary_method)
                v = {_k: [_v[0][_i] if len(_v[0]) > _i else _v[0][_i % len(_v[0])]] for _k, _v in v.items()}
            else:
                raise ValueError(f'unknown method: {opt.rewrite_dictionary_method}')

            logging.info(f'Rewriting {len(v)}/{len(dataset[k])} texts')
            if opt.add_rewrite_text:
                logging.info(f"adding rewritten data ({k}): {len(dataset[k])}")
                tmp_data = dataset[k]
                for i in tmp_data:
                    if i['text'] in v:
                        for augmented_texts in v[i['text']]:
                            dataset[k] = dataset[k].add_item({'text': augmented_texts, 'label': i['label']})
                logging.info(f"final training data: {len(dataset[k])}")
            else:
                dataset[k] = dataset[k].map(lambda x: {
                    'text': x['text'] if x['text'] not in v else v[x['text']][0],
                    'label': x['label']
                })
    network = internet_connection()
    # setup model
    tokenizer = AutoTokenizer.from_pretrained(opt.model, local_files_only=not network)
    model = AutoModelForSequenceClassification.from_pretrained(
        opt.model, num_labels=dataset['train'].features['label'].num_classes, local_files_only=not network)
    tokenized_datasets = dataset.map(
        lambda x: tokenizer(x["text"], padding="max_length", truncation=True, max_length=opt.seq_length),
        batched=True)
    # setup metrics
    compute_metric_search, compute_metric_all = get_metrics()

    if not opt.skip_train:
        # setup trainer
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=opt.output_dir,
                evaluation_strategy="steps",
                eval_steps=opt.eval_step,
                seed=opt.random_seed
            ),
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            compute_metrics=compute_metric_search,
            model_init=lambda x: AutoModelForSequenceClassification.from_pretrained(
                opt.model, return_dict=True, num_labels=dataset['train'].features['label'].num_classes)
        )
        # parameter search
        if PARALLEL:
            best_run = trainer.hyperparameter_search(
                hp_space=lambda x: {
                    "learning_rate": tune.loguniform(1e-6, 1e-4),
                    "num_train_epochs": tune.choice(list(range(1, 6))),
                    "per_device_train_batch_size": tune.choice([4, 8, 16, 32, 64]),
                },
                local_dir=RAY_RESULTS, direction="maximize", backend="ray", n_trials=opt.n_trials,
                resources_per_trial={'cpu': multiprocessing.cpu_count(), "gpu": torch.cuda.device_count()},

            )
        else:
            best_run = trainer.hyperparameter_search(
                hp_space=lambda x: {
                    "learning_rate": tune.loguniform(1e-6, 1e-4),
                    "num_train_epochs": tune.choice(list(range(1, 6))),
                    "per_device_train_batch_size": tune.choice([4, 8, 16, 32, 64]),
                },
                local_dir=RAY_RESULTS, direction="maximize", backend="ray", n_trials=opt.n_trials
            )
        # finetuning
        for n, v in best_run.hyperparameters.items():
            setattr(trainer.args, n, v)
        trainer.train()
        trainer.save_model(pj(opt.output_dir, 'best_model'))
        best_model_path = pj(opt.output_dir, 'best_model')
    else:
        best_model_path = opt.output_dir

    # evaluation
    model = AutoModelForSequenceClassification.from_pretrained(
        best_model_path,
        num_labels=dataset['train'].features['label'].num_classes,
        local_files_only=not network)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=opt.output_dir,
            evaluation_strategy="no",
            seed=opt.random_seed
        ),
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metric_all,
        model_init=lambda x: AutoModelForSequenceClassification.from_pretrained(
            opt.model, return_dict=True, num_labels=dataset['train'].features['label'].num_classes)
    )
    summary_file = pj(opt.output_dir, opt.summary_file)
    if not opt.skip_eval:
        result = {f'test/{k}': v for k, v in trainer.evaluate().items()}
        logging.info(json.dumps(result, indent=4))
        with open(summary_file, 'w') as f:
            json.dump(result, f)

    if opt.push_to_hub:
        assert opt.hf_organization is not None, f'specify hf organization `--hf-organization`'
        assert opt.model_alias is not None, f'specify hf organization `--model-alias`'
        url = create_repo(opt.model_alias, organization=opt.hf_organization, exist_ok=True)
        # if not opt.skip_train:
        args = {"use_auth_token": opt.use_auth_token, "repo_url": url, "organization": opt.hf_organization}
        trainer.model.push_to_hub(opt.model_alias, **args)
        tokenizer.push_to_hub(opt.model_alias, **args)
        if os.path.exists(summary_file):
            shutil.copy2(summary_file, opt.model_alias)
        os.system(
            f"cd {opt.model_alias} && git lfs install && git add . && git commit -m 'model update' && git push && cd ../")
        shutil.rmtree(f"{opt.model_alias}")  # clean up the cloned repo


if __name__ == '__main__':
    main()
