### GREEDY SEARCH
wandb offline
export WANDB_DISABLED='true'
export PARALLEL='0'
export RAY_RESULTS='ray_results'

main () {
  DATA=${1}
  MODEL=${2}
  BATCH=64
  CHUNK=250

  if [ "${DATA}" = "citation_intent" ]
    then
      MAX_LENGTH=128
  elif [ "${DATA}" = 'sciie' ]
    then
      MAX_LENGTH=128
  elif [ "${DATA}" = 'rct_sample' ]
    then
      MAX_LENGTH=256
  else
    MAX_LENGTH=64
  fi

  SPLIT='train'
  m3-rewrite -b 64 -m "${MODEL}" --topk-buffer 1 -n 10 -k 1 -l ${MAX_LENGTH} -b ${BATCH} -d m3/multi_domain_document_classification --dataset-name "${DATA}" -s ${SPLIT} --dataset-column text -e "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top1.iteration10/${SPLIT}.json" -c ${CHUNK}
# fine-tuning on m3 dataset
#  python lm_finetuning.py -m "${MODEL}" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v2-greedy" --summary-file 'metric_summary.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top1.iteration1" --rewrite-dictionary-split 'train' --add-rewrite-text --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v2-greedy" --rewrite-dictionary-method 'best'
#  python lm_finetuning.py -m "${MODEL}" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v3-greedy" --summary-file 'metric_summary.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top1.iteration1" --rewrite-dictionary-split 'train' --add-rewrite-text --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v3-greedy" --rewrite-dictionary-method 'largest_diff'
}

# rct_sample must be missed!!!!!
MODEL="albert-base-v2"
main ${DATA} ${MODEL}
DATA="rct_sample"
MODEL="roberta-base"
main ${DATA} ${MODEL}
for DATA in "amcd" "chemprot" "citation_intent" "rct_sample" "sciie" "tweet_eval_irony" "tweet_eval_hate" "tweet_eval_emotion"
for DATA in "citation_intent" "sciie" "tweet_eval_irony" "tweet_eval_hate" "tweet_eval_emotion"
for DATA in "rct_sample"
do
  main ${DATA} ${MODEL}
done




