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
- citation_intent
```shell
DATA='citation_intent'
MODEL='albert-base-v2'  #  distilbert-base-uncased, distilbert-base-cased
MAX_LENGTH=128
BATCH=512
ORG='asahi417'
```
- amcd
```shell
DATA='amcd'
MODEL='albert-base-v2'  #  distilbert-base-uncased, distilbert-base-cased
MAX_LENGTH=64
BATCH=512
ORG='asahi417'
```

### Run
 
```shell
# generate inputs based on M3
for SPLIT in 'train' 'validation' 'test'
do
  m3-rewrite -m ${MODEL} -n 10 -k 10 -l ${MAX_LENGTH} -b ${BATCH} -d asahi417/multi_domain_document_classification --dataset-name ${DATA} -s ${SPLIT} --dataset-column text \
  -e "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10/${SPLIT}.json"
done

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
```