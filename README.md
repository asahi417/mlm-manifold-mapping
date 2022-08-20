# M3: Masked Language Model Manifold Mapping

## Sample
```shell
m3-rewriter -i "AHH i'm so HAPPY." "I just found my ipod. God is sooo good to me" -n 3 -k 2 
```

```shell
m3-rewriter -f "tests/sample_sentence.txt" -n 3 -k 2 
```

## M3 Experiments
###  Config
- model
```shell
MODEL='albert-base-v2'
BATCH=128
CHUNK=250

MODEL='roberta-base'
BATCH=64
CHUNK=250
```

- dataset
```shell
DATA='citation_intent'
MAX_LENGTH=128

DATA='amcd'
MAX_LENGTH=64

DATA='sciie'
MAX_LENGTH=128

DATA='chemprot'
MAX_LENGTH=128

DATA='tweet_eval_emotion'
MAX_LENGTH=64

DATA='tweet_eval_hate'
MAX_LENGTH=64

DATA='tweet_eval_irony'
MAX_LENGTH=64

DATA='yelp_review'
MAX_LENGTH=256
```


### Generate M3 Data
```shell
# generate inputs based on M3
for SPLIT in 'train' 'validation' 'test'
do
  m3-rewrite -m ${MODEL} -n 10 -k 10 -l ${MAX_LENGTH} -b ${BATCH} -d m3/multi_domain_document_classification --dataset-name ${DATA} -s ${SPLIT} --dataset-column text -e "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10/${SPLIT}.json" -c ${CHUNK}
done
```

### Finetuning
- To turn off wandb
```shell
wandb offline
export WANDB_DISABLED='true'
```

- experiment v1
```shell
# fine-tuning on vanilla dataset
python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.vanilla" --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-vanilla" --summary-file 'metric_summary.json'  
# fine-tuning on m3 dataset
python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.replace" --summary-file 'metric_summary.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'train' 'validation' --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-replace"
# fine-tuning on m3 dataset (concatenation)
python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add" --summary-file 'metric_summary.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'train' 'validation' --add-rewrite-text --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add"

# evaluate m3 on vanilla dataset fine-tuned model
python lm_finetuning.py -m "m3/m3-experiment-${MODEL}-${DATA//_/-}-vanilla" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.vanilla/best_model" --skip-train --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-vanilla" --summary-file 'metric_summary.edit.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'test'  
# evaluate m3 on m3 dataset fine-tuned model 
python lm_finetuning.py -m "m3/m3-experiment-${MODEL}-${DATA//_/-}-replace" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.replace/best_model" --skip-train --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-replace" --summary-file 'metric_summary.edit.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'test'
# evaluate m3 on m3 dataset fine-tuned model (concatenation)
python lm_finetuning.py -m "m3/m3-experiment-${MODEL}-${DATA//_/-}-add" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add/best_model" --skip-train --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add" --summary-file 'metric_summary.edit.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'test'
```

- experiment v2: no edit on validation

```shell
# fine-tuning on m3 dataset
python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.replace-v2" --summary-file 'metric_summary.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'train' --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-replace-v2"
# fine-tuning on m3 dataset (concatenation)
python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v2" --summary-file 'metric_summary.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'train' --add-rewrite-text --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v2"
# evaluate m3 on m3 dataset fine-tuned model
python lm_finetuning.py -m "m3/m3-experiment-${MODEL}-${DATA//_/-}-replace-v2" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.replace-v2/best_model" --skip-train --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-replace-v2" --summary-file 'metric_summary.edit.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'test'
# evaluate m3 on m3 dataset fine-tuned model (concatenation)
python lm_finetuning.py -m "m3/m3-experiment-${MODEL}-${DATA//_/-}-add-v2" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v2/best_model" --skip-train --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v2" --summary-file 'metric_summary.edit.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'test'
```

- experiment v3: best improved edit

```shell
# fine-tuning on m3 dataset (concatenation)
python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v3" --summary-file 'metric_summary.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'train' --add-rewrite-text --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --rewrite-dictionary-method 'largest_diff'
# evaluate m3 on m3 dataset fine-tuned model (concatenation)
python lm_finetuning.py -m "m3/m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v3/best_model" --skip-train --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --summary-file 'metric_summary.edit.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'test' --rewrite-dictionary-method 'largest_diff'
```

# MISC

```shell
roberta: only v2
citation_intent: running on hawk
amcd: running on hawk
chemprot: running on hawk


TODO:
- RoBERTa: v2
- AlBERT, RoBERTa: v3
- M3: Roberta: yelp (cardiff_nlp), sciie_train (stone)
- v3: cardiff_nlp
 
```

```shell
for DATA in "chemprot" "citation_intent" "sciie" "amcd" "tweet_eval_emotion" "tweet_eval_irony" "tweet_eval_hate"
do
  for TYPE in "vanilla" "add" "replace"
  do
    git clone "https://huggingface.co/m3/m3-experiment-${MODEL}-${DATA//_/-}-${TYPE}"
    python lm_finetuning.py -m "m3/m3-experiment-${MODEL}-${DATA//_/-}-${TYPE}" --dataset-name "${DATA}" -o "m3-experiment-${MODEL}-${DATA//_/-}-${TYPE}" --skip-train --summary-file 'metric_summary.json'
#    python lm_finetuning.py -m "m3/m3-experiment-${MODEL}-${DATA//_/-}-${TYPE}" --dataset-name "${DATA}" -o "m3-experiment-${MODEL}-${DATA//_/-}-${TYPE}" --skip-train --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-${TYPE}" --summary-file 'metric_summary.edit.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'test'
    cd "m3-experiment-${MODEL}-${DATA//_/-}-${TYPE}"
    rm -rf runs
    git lfs install && git add . && git commit -m 'model update' && git push && cd ../ 
    rm -rf "m3-experiment-${MODEL}-${DATA//_/-}-${TYPE}"
  done
done
```

```shell
wandb offline
export WANDB_DISABLED='true'

python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.replace-v2" --summary-file 'metric_summary.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'train'
python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v2" --summary-file 'metric_summary.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'train' --add-rewrite-text
```


```shell

export PARALLEL=0
wandb offline
export WANDB_DISABLED='true'

python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.vanilla" --skip-eval --skip-train --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-vanilla"
python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add" --skip-eval --skip-train --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add"
python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v2" --skip-eval --skip-train --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v2"
python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.replace" --skip-eval --skip-train --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-replace"
python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.replace-v2" --skip-eval --skip-train --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-replace-v2"

for MODEL in 'albert-base-v2' 'roberta-base'
do
    for DATA in "chemprot" "citation_intent" "sciie" "amcd" "tweet_eval_emotion" "tweet_eval_irony" "tweet_eval_hate"
    for DATA in "chemprot" "citation_intent" "sciie" "amcd" "tweet_eval_emotion"
    for DATA in "citation_intent" "sciie" "amcd"
#    for DATA in "chemprot" "tweet_eval_emotion"
    do 
#        python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.vanilla" --skip-train
#        python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.vanilla" --skip-train --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'test' --summary-file 'metric_summary.edit.json'
        
        python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add" --skip-train
        python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.replace" --skip-train        
        python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add" --skip-train --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'test' --summary-file 'metric_summary.edit.json'
        python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.replace" --skip-train --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'test' --summary-file 'metric_summary.edit.json'
        python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v2/best_model" --skip-train
        python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.replace-v2" --skip-train
        python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v2/best_model" --skip-train --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'test' --summary-file 'metric_summary.edit.json'
        python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.replace-v2/best_model" --skip-train --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'test' --summary-file 'metric_summary.edit.json'        
    done
done




for MODEL in 'albert-base-v2' 'roberta-base'
do
    for DATA in "chemprot" "citation_intent" "sciie" "amcd" "tweet_eval_irony" "tweet_eval_hate" "tweet_eval_emotion"
#    for DATA in "chemprot" "tweet_eval_emotion"
    for DATA in "citation_intent" "sciie" "amcd" 
    do 
#        python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.vanilla" --skip-eval --skip-train --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-vanilla"
        python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add" --skip-eval --skip-train --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add"
        python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.replace" --skip-eval --skip-train --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-replace"
        python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add" --skip-eval --skip-train --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add" --summary-file 'metric_summary.edit.json'
        python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.replace" --skip-eval --skip-train --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-replace" --summary-file 'metric_summary.edit.json'
        
        python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v2/best_model" --skip-eval --skip-train --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v2"
        python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.replace-v2/best_model" --skip-eval --skip-train --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-replace-v2"
        python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v2/best_model" --skip-eval --skip-train --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v2" --summary-file 'metric_summary.edit.json'
        python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.replace-v2/best_model" --skip-eval --skip-train --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-replace-v2" --summary-file 'metric_summary.edit.json'
    done
done 
```
