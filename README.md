# M3: Masked Language Model Manifold Mapping

## Sample
```shell
m3-rewriter -i "AHH i'm so HAPPY." "I just found my ipod. God is sooo good to me" -n 3 -k 2 
```

```shell
m3-rewriter -f "tests/sample_sentence.txt" -n 3 -k 2 
```

## Finetuning with M3
- Generate new inputs based on M3 
```shell
DATA='citation_intent'
MODEL='albert-base-v2'  #  distilbert-base-uncased, distilbert-base-cased
MAX_LENGTH=128
BATCH=512
ORG='asahi417'
# generate inputs based on M3
for SPLIT in 'train' 'validation' 'test'
do
  m3-rewrite -m ${MODEL} -n 10 -k 10 -l ${MAX_LENGTH} -b ${BATCH} -d asahi417/multi_domain_document_classification --dataset-name ${DATA} -s ${SPLIT} --dataset-column text \
  -e "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10/${SPLIT}.json"
done

# vanilla fine-tuning
python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}" --push-to-hub --hf-organization ${ORG} -a "m3-experiment-${MODEL}-${DATA}-vanilla" --summary-file 'metric_summary.json'  

python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_result/${MODEL}.${DATA}" -t 5 \
--push-to-hub --hf-organization asahi417 -a "m3-experiment-${MODEL}-${DATA}-replace" \
--rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10"
```

```shell
DATA='amcd'
MAX_LENGTH=64
MODEL='albert-base-v2'  #  distilbert-base-uncased, distilbert-base-cased
BATCH=512
ORG='asahi417'
# generate inputs based on M3
for SPLIT in 'train' 'validation' 'test'
do
  m3-rewrite -m ${MODEL} -n 10 -k 10 -l ${MAX_LENGTH} -b ${BATCH} -d asahi417/multi_domain_document_classification --dataset-name ${DATA} -s ${SPLIT} --dataset-column text \
  -e "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10/${SPLIT}.json"
done

DATA='citation_intent'
MAX_LENGTH=128
MODEL='albert-base-v2'  #  distilbert-base-uncased, distilbert-base-cased
BATCH=512
ORG='asahi417'
# generate inputs based on M3
for SPLIT in 'train' 'validation' 'test'
do
  m3-rewrite -m ${MODEL} -n 10 -k 10 -l ${MAX_LENGTH} -b ${BATCH} -d asahi417/multi_domain_document_classification --dataset-name ${DATA} -s ${SPLIT} --dataset-column text \
  -e "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10/${SPLIT}.json"
done
```