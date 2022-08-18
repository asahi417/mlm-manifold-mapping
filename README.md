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
BATCH=32
CHUNK=250
```

- dataset
```shell
DATA='citation_intent'
MAX_LENGTH=128

# v2
DATA='amcd'
MAX_LENGTH=64

# v2
DATA='sciie'
MAX_LENGTH=128


DATA='chemprot'
MAX_LENGTH=128

# all
DATA='yelp_review'
MAX_LENGTH=256

DATA='tweet_eval_emotion'
MAX_LENGTH=64

# all
DATA='tweet_eval_hate'
MAX_LENGTH=64

DATA='tweet_eval_irony'
MAX_LENGTH=64
```


### Run Experiments 
- generate m3 data
```shell
# generate inputs based on M3
#for SPLIT in 'train' 'validation'
for SPLIT in 'train' 'validation' 'test'
do
  m3-rewrite -m ${MODEL} -n 10 -k 10 -l ${MAX_LENGTH} -b ${BATCH} -d asahi417/multi_domain_document_classification --dataset-name ${DATA} -s ${SPLIT} --dataset-column text \
  -e "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10/${SPLIT}.json" -c ${CHUNK}
done
```

- finetuning

```shell
# To turn off wandb
# wandb offline
# export WANDB_DISABLED='true'

ORG='asahi417'
# fine-tuning on vanilla dataset
python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.vanilla" \
--push-to-hub --hf-organization ${ORG} -a "m3-experiment-${MODEL}-${DATA//_/-}-vanilla" --summary-file 'metric_summary.json'  
# fine-tuning on m3 dataset
python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.replace" \
--push-to-hub --hf-organization ${ORG} -a "m3-experiment-${MODEL}-${DATA//_/-}-replace" --summary-file 'metric_summary.json' \
--rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'train' 'validation'
# fine-tuning on m3 dataset (concatenation)
python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add" \
--push-to-hub --hf-organization ${ORG} -a "m3-experiment-${MODEL}-${DATA//_/-}-add" --summary-file 'metric_summary.json' \
--rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'train' 'validation' --add-rewrite-text

# fine-tuning on m3 dataset (validation on original data)
python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.replace-v2" \
--push-to-hub --hf-organization ${ORG} -a "m3-experiment-${MODEL}-${DATA//_/-}-replace-v2" --summary-file 'metric_summary.json' \
--rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'train'
# fine-tuning on m3 dataset (concatenation) (validation on original data)
python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v2" \
--push-to-hub --hf-organization ${ORG} -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v2" --summary-file 'metric_summary.json' \
--rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'train' --add-rewrite-text


# evaluate m3 on vanilla dataset fine-tuned model
python lm_finetuning.py -m "${ORG}/m3-experiment-${MODEL}-${DATA//_/-}-vanilla" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.vanilla" --skip-train \
--push-to-hub --hf-organization ${ORG} -a "m3-experiment-${MODEL}-${DATA//_/-}-vanilla" --summary-file 'metric_summary.edit.json' \
--rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'test'  
# evaluate m3 on m3 dataset fine-tuned model 
python lm_finetuning.py -m "${ORG}/m3-experiment-${MODEL}-${DATA//_/-}-replace" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.replace" --skip-train \
--push-to-hub --hf-organization ${ORG} -a "m3-experiment-${MODEL}-${DATA//_/-}-replace" --summary-file 'metric_summary.edit.json' \
--rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'test'
# evaluate m3 on m3 dataset fine-tuned model (concatenation)
python lm_finetuning.py -m "${ORG}/m3-experiment-${MODEL}-${DATA//_/-}-add" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add" --skip-train \
--push-to-hub --hf-organization ${ORG} -a "m3-experiment-${MODEL}-${DATA//_/-}-add" --summary-file 'metric_summary.edit.json' \
--rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'test'
# evaluate m3 on m3 dataset fine-tuned model (validation on original data)
python lm_finetuning.py -m "${ORG}/m3-experiment-${MODEL}-${DATA//_/-}-replace-v2" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.replace-v2" --skip-train \
--push-to-hub --hf-organization ${ORG} -a "m3-experiment-${MODEL}-${DATA//_/-}-replace-v2" --summary-file 'metric_summary.edit.json' \
--rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'test'
# evaluate m3 on m3 dataset fine-tuned model (concatenation) (validation on original data)
python lm_finetuning.py -m "${ORG}/m3-experiment-${MODEL}-${DATA//_/-}-add-v2" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v2" --skip-train \
--push-to-hub --hf-organization ${ORG} -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v2" --summary-file 'metric_summary.edit.json' \
--rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'test'
```


```shell
wandb offline
export WANDB_DISABLED='true'

python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.vanilla" --summary-file 'metric_summary.json'  


python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.replace" --summary-file 'metric_summary.json' \
--rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'train' 'validation'
# fine-tuning on m3 dataset (concatenation)
python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add" --summary-file 'metric_summary.json' \
--rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'train' 'validation' --add-rewrite-text

# fine-tuning on m3 dataset (validation on original data)
python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.replace-v2" --summary-file 'metric_summary.json' \
--rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'train'
# fine-tuning on m3 dataset (concatenation) (validation on original data)
python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v2" --summary-file 'metric_summary.json' \
--rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'train' --add-rewrite-text
```


```shell
ORG='asahi417'
python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.vanilla" --skip-eval --skip-train --push-to-hub --hf-organization ${ORG} -a "m3-experiment-${MODEL}-${DATA//_/-}-vanilla"
python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add" --skip-eval --skip-train --push-to-hub --hf-organization ${ORG} -a "m3-experiment-${MODEL}-${DATA//_/-}-add"
python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v2" --skip-eval --skip-train --push-to-hub --hf-organization ${ORG} -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v2"
python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.replace" --skip-eval --skip-train --push-to-hub --hf-organization ${ORG} -a "m3-experiment-${MODEL}-${DATA//_/-}-replace"
python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.replace-v2" --skip-eval --skip-train --push-to-hub --hf-organization ${ORG} -a "m3-experiment-${MODEL}-${DATA//_/-}-replace-v2" 
```