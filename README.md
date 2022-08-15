# M3: Masked Language Model Manifold Mapping


```shell
m3-rewriter -i "AHH i'm so HAPPY." "I just found my ipod. God is sooo good to me" -n 3 -k 2 
```

```shell
m3-rewriter -f "tests/sample_sentence.txt" -n 3 -k 2 
```

```shell
export DATA='citation_intent'
export MODEL='albert-base-v2'
export SPLIT='train'
m3-rewrite -m ${MODEL} -n 5 -k 5 -l 64 -b 64 -d asahi417/multi_domain_document_classification \
 --dataset-name ${DATA} -s ${SPLIT} --dataset-column text -e "result_m3/${DATA}/${SPLIT}.5.5.64"
export SPLIT='validation'
m3-rewrite -m ${MODEL} -n 5 -k 5 -l 128 -b 512 -d asahi417/multi_domain_document_classification \
 --dataset-name ${DATA} -s ${SPLIT} --dataset-column text -e "result_m3/${DATA}/${SPLIT}.5.5"
export SPLIT='test'
m3-rewrite -m ${MODEL} -n 5 -k 5 -l 64 -b 512 -d asahi417/multi_domain_document_classification \
 --dataset-name ${DATA} -s ${SPLIT} --dataset-column text -e "result_m3/${DATA}/${SPLIT}.5.5" 
```

```shell
#DATA="rct-sample"
DATA="citation_intent"
DATA="sciie"
DATA="hyperpartisan_news"
mkdir "${DATA}"
curl -Lo "${DATA}/train.jsonl" "https://allennlp.s3-us-west-2.amazonaws.com/dont_stop_pretraining/data/${DATA}/train.jsonl"
curl -Lo "${DATA}/dev.jsonl" "https://allennlp.s3-us-west-2.amazonaws.com/dont_stop_pretraining/data/${DATA}/dev.jsonl"
curl -Lo "${DATA}/test.jsonl" "https://allennlp.s3-us-west-2.amazonaws.com/dont_stop_pretraining/data/${DATA}/test.jsonl"
```