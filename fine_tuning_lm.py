import argparse
import numpy as np
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer


def get_metrics():
    metric_accuracy = load_metric("accuracy")
    metric_f1 = load_metric("f1")

    def compute_f1_micro(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric_f1.compute(predictions=predictions, references=labels, average='micro')

    def compute_f1_macro(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric_f1.compute(predictions=predictions, references=labels, average='macro')

    def compute_accuracy(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric_accuracy.compute(predictions=predictions, references=labels)

    return compute_f1_micro, compute_f1_macro, compute_accuracy


def main():
    parser = argparse.ArgumentParser(description='Fine-tuning language model.')
    # config
    parser.add_argument('-m', '--model', help='transformer LM', default='albert-base-v2', type=str)
    parser.add_argument('-d', '--dataset', help='', default='asahi417/multi_domain_document_classification', type=str)
    parser.add_argument('--dataset-name', help='huggingface dataset name', default='citation_intent', type=str)
    parser.add_argument('-b', '--batch-size', help='batch size', default=64, type=int)
    parser.add_argument('-e', '--epoch', help='epoch', default=10, type=int)
    parser.add_argument('-l', '--seq-length', help='', default=128, type=int)
    parser.add_argument('--random-seed', help='', default=42, type=int)
    parser.add_argument('--eval-step', help='', default=10, type=int)
    parser.add_argument('-o', '--output-dir', help='Directory to output', required=True, type=str)
    parser.add_argument('--hyperparameter-search', action='store_true')
    # input
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
    # setup metrics
    compute_f1_micro, compute_f1_macro, compute_accuracy = get_metrics()
    # trainer
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=opt.output_dir,
            evaluation_strategy="steps",
            num_train_epochs=opt.epoch,
            eval_steps=opt.eval_step,
            seed=opt.random_seed,
            per_device_train_batch_size=opt.batch_size
        ),
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        compute_metrics=compute_f1_micro,
        model_init=lambda x: AutoModelForSequenceClassification.from_pretrained(
            opt.model, return_dict=True, num_labels=dataset['train'].features['label'].num_classes)
    )
    if opt.hyperparameter_search:
        trainer.hyperparameter_search(
            direction="maximize",
            backend="ray",
            n_trials=10  # number of trials
        )
    else:
        trainer.train()
    trainer.evaluate()

if __name__ == '__main__':
    main()
