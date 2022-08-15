"""
- TrainingArguments
https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments
https://huggingface.co/transformers/v3.1.0/_modules/transformers/trainer_utils.html
https://github.com/huggingface/notebooks/blob/main/examples/text_classification.ipynb

MODEL='albert-base-v2'  #  distilbert-base-uncased, distilbert-base-cased
DATA='citation_intent'
python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_result/${MODEL}.${DATA}" -t 5 \
--push-to-hub --hf-organization asahi417 -a tmp
"""
import argparse
import json
import logging
import os
import shutil
from os.path import join as pj

import numpy as np
from huggingface_hub import create_repo
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from ray import tune
logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def main():
    parser = argparse.ArgumentParser(description='Fine-tuning language model.')
    parser.add_argument('-m', '--model', help='transformer LM', default='albert-base-v2', type=str)
    parser.add_argument('-d', '--dataset', help='', default='asahi417/multi_domain_document_classification', type=str)
    parser.add_argument('--dataset-name', help='huggingface dataset name', default='citation_intent', type=str)
    parser.add_argument('-l', '--seq-length', help='', default=128, type=int)
    parser.add_argument('--random-seed', help='', default=42, type=int)
    parser.add_argument('--eval-step', help='', default=50, type=int)
    parser.add_argument('-o', '--output-dir', help='Directory to output', default='tmp', type=str)
    parser.add_argument('-t', '--n-trials', default=5, type=int)
    parser.add_argument('--push-to-hub', action='store_true')
    parser.add_argument('--use-auth-token', action='store_true')
    parser.add_argument('--hf-organization', default=None, type=str)
    parser.add_argument('-a', '--model-alias', help='', default=None, type=str)
    opt = parser.parse_args()

    # setup data
    dataset = load_dataset(opt.dataset, opt.dataset_name)
    # setup model
    tokenizer = AutoTokenizer.from_pretrained(opt.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        opt.model, num_labels=dataset['train'].features['label'].num_classes)
    tokenized_datasets = dataset.map(
        lambda x: tokenizer(x["text"], padding="max_length", truncation=True, max_length=opt.seq_length),
        batched=True)
    # setup metric
    metric_accuracy = load_metric("accuracy")
    metric_f1 = load_metric("f1")

    ##########################
    # HYPER-PARAMETER SEARCH #
    ##########################

    def compute_metric_search(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric_f1.compute(predictions=predictions, references=labels, average='micro')

    # parameter search
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

    best_run = trainer.hyperparameter_search(
        hp_space=lambda x: {
            "learning_rate": tune.loguniform(1e-6, 1e-4),
            "num_train_epochs": tune.choice(list(range(1, 6))),
            "per_device_train_batch_size": tune.choice([4, 8, 16, 32, 64]),
        },
        local_dir="ray_results",
        direction="maximize",
        backend="ray",
        n_trials=opt.n_trials  # number of trials
    )

    ##############
    # FINETUNING #
    ##############

    def compute_metric_all(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            'f1': metric_f1.compute(predictions=predictions, references=labels, average='micro')['f1'],
            'f1_macro': metric_f1.compute(predictions=predictions, references=labels, average='macro')['f1'],
            'accuracy': metric_accuracy.compute(predictions=predictions, references=labels)['accuracy']
        }

    for n, v in best_run.hyperparameters.items():
        setattr(trainer.args, n, v)

    trainer_output = trainer.train()
    result = trainer_output.metrics
    trainer.compute_metrics = compute_metric_all
    result.update({
        f'valid/{k}': v for k, v in trainer.evaluate(eval_dataset=tokenized_datasets['validation']).items()
    })
    result.update({
        f'test/{k}': v for k, v in trainer.evaluate(eval_dataset=tokenized_datasets['test']).items()
    })
    logging.info(json.dumps(result, indent=4))
    with open(pj(opt.output_dir, 'metric_summary.json'), 'w') as f:
        json.dump(result, f)

    if opt.push_to_hub:
        ###############
        # PUSH TO HUB #
        ###############
        assert opt.hf_organization is not None, f'specify hf organization `--hf-organization`'
        assert opt.model_alias is not None, f'specify hf organization `--model-alias`'
        url = create_repo(opt.model_alias, organization=opt.hf_organization, exist_ok=True)
        args = {"use_auth_token": opt.use_auth_token, "repo_url": url, "organization": opt.hf_organization}
        trainer.model.push_to_hub(opt.model_alias, **args)
        tokenizer.push_to_hub(opt.model_alias, **args)
        shutil.copy2(pj(opt.output_dir, 'metric_summary.json'), opt.model_alias)
        os.system(
            f"cd {opt.model_alias} && git lfs install && git add . && git commit -m 'model update' && git push && cd ../")
        shutil.rmtree(f"{opt.model_alias}")  # clean up the cloned repo


if __name__ == '__main__':
    main()
