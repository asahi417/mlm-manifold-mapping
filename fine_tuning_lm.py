import argparse
import numpy as np
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer


def main():
    parser = argparse.ArgumentParser(description='Fine-tuning language model.')
    # config
    parser.add_argument('-m', '--model', help='transformer LM', default='albert-base-v2', type=str)
    parser.add_argument('-d', '--dataset', help='', default='asahi417/multi_domain_document_classification', type=str)
    parser.add_argument('--dataset-name', help='huggingface dataset name', default='citation_intent', type=str)
    parser.add_argument('-b', '--batch-size', help='Batch size', default=512, type=int)
    parser.add_argument('--random-seed', help='', default=42, type=int)
    parser.add_argument('-o', '--output-dir', help='Directory to output', default='.', type=str)
    # input
    opt = parser.parse_args()

    # set up data
    dataset = load_dataset(opt.dataset, opt.dataset_name)
    # set up model
    tokenizer = AutoTokenizer.from_pretrained(opt.model)
    model = AutoModelForSequenceClassification.from_pretrained(opt.model, num_labels=5)

    def model_init():
        return AutoModelForSequenceClassification.from_pretrained(opt.model, return_dict=True)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    lm_name = "albert-base-v2"
    dataset = "yelp_review_full"
    dataset = load_dataset(dataset)



tokenized_datasets = dataset.map(tokenize_function, batched=True)
small_train_dataset = tokenized_datasets["train"].shuffle(seed=opt.random_seed).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=opt.random_seed).select(range(1000))


metric = load_metric("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)




training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="steps", eval_steps=500)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
    model_init=model_init
)
# trainer.train()
trainer.hyperparameter_search(
    direction="maximize",
    backend="ray",
    n_trials=10  # number of trials
)