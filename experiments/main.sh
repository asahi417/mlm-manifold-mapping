
wandb offline
export WANDB_DISABLED='true'


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

#  python lm_finetuning.py -m "${MODEL}" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.replace-v3" --summary-file 'metric_summary.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'train' --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-replace-v3" --rewrite-dictionary-method 'largest_diff'

  SPLIT='train'
  m3-rewrite -m "${MODEL}" -n 10 -k 10 -l ${MAX_LENGTH} -b ${BATCH} -d m3/multi_domain_document_classification --dataset-name "${DATA}" -s ${SPLIT} --dataset-column text -e "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10/${SPLIT}.json" -c ${CHUNK}

  # fine-tuning on vanilla dataset
  python lm_finetuning.py -m "${MODEL}" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.vanilla" --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-vanilla" --summary-file 'metric_summary.json'
  # fine-tuning on m3 dataset
  python lm_finetuning.py -m "${MODEL}" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v2" --summary-file 'metric_summary.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'train' --add-rewrite-text --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v2" --rewrite-dictionary-method 'best'
  python lm_finetuning.py -m "${MODEL}" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v3" --summary-file 'metric_summary.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'train' --add-rewrite-text --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --rewrite-dictionary-method 'largest_diff'
}


for DATA in "amcd" "chemprot" "citation_intent" "rct_sample" "sciie" "tweet_eval_irony" "tweet_eval_hate" "tweet_eval_emotion"
do
  for MODEL in "albert-base-v2" "roberta-base"
  do
    main ${DATA} ${MODEL}
  done
done

for DATA in "tweet_eval_hate" "tweet_eval_emotion"
do
  main ${DATA} ${MODEL}
done



### GREEDY SEARCH
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
  m3-rewrite -m "${MODEL}" -n 1 -k 1 -l ${MAX_LENGTH} -b ${BATCH} -d m3/multi_domain_document_classification --dataset-name "${DATA}" -s ${SPLIT} --dataset-column text -e "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top1.iteration1/${SPLIT}.json" -c ${CHUNK}
# fine-tuning on m3 dataset
  python lm_finetuning.py -m "${MODEL}" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v2-greedy" --summary-file 'metric_summary.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top1.iteration1" --rewrite-dictionary-split 'train' --add-rewrite-text --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v2-greedy" --rewrite-dictionary-method 'best'
  python lm_finetuning.py -m "${MODEL}" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v3-greedy" --summary-file 'metric_summary.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top1.iteration1" --rewrite-dictionary-split 'train' --add-rewrite-text --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v3-greedy" --rewrite-dictionary-method 'largest_diff'
}


for DATA in "amcd" "chemprot" "citation_intent" "rct_sample" "sciie" "tweet_eval_irony" "tweet_eval_hate" "tweet_eval_emotion"
do
  for MODEL in "albert-base-v2" "roberta-base"
  do
    main ${DATA} ${MODEL}
  done
done

for DATA in "tweet_eval_hate" "tweet_eval_emotion"
do
  main ${DATA} ${MODEL}
done
