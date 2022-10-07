# main model training with m3
wandb offline
export WANDB_DISABLED='true'
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
#  m3-rewrite -m "${MODEL}" -n 10 -k 10 -l ${MAX_LENGTH} -b ${BATCH} -d m3/multi_domain_document_classification --dataset-name "${DATA}" -s ${SPLIT} --dataset-column text -e "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10/${SPLIT}.json" -c ${CHUNK}
#  # fine-tuning on vanilla dataset
#  python lm_finetuning.py -m "${MODEL}" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.vanilla" --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-vanilla" --summary-file 'metric_summary.json'
#  # fine-tuning on m3 dataset
#  python lm_finetuning.py -m "${MODEL}" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v2" --summary-file 'metric_summary.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'train' --add-rewrite-text --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v2" --rewrite-dictionary-method 'best'
#  python lm_finetuning.py -m "${MODEL}" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v3" --summary-file 'metric_summary.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'train' --add-rewrite-text --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --rewrite-dictionary-method 'largest_diff'
#  python lm_finetuning.py -m "${MODEL}" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v4" --summary-file 'metric_summary.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'train' --add-rewrite-text --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v4" --rewrite-dictionary-method 'all'
  python lm_finetuning.py -m "${MODEL}" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v5" --summary-file 'metric_summary.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'train' --add-rewrite-text --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v5" --rewrite-dictionary-method '1'
}


for MODEL in "albert-base-v2" "roberta-base"
do
#  for DATA in "amcd" "chemprot" "citation_intent" "rct_sample" "sciie" "tweet_eval_irony" "tweet_eval_hate" "tweet_eval_emotion"
  for DATA in "amcd" "citation_intent" "rct_sample" "tweet_eval_irony" "tweet_eval_hate" "tweet_eval_emotion"
  do
    main ${DATA} ${MODEL}
    rm -rf ${RAY_RESULTS}
  done
done



MODEL="roberta-base"
export RAY_RESULTS="ray_results_${MODEL}"
DATA="tweet_eval_hate"
main ${DATA} ${MODEL}
rm -rf ${RAY_RESULTS}


MODEL="albert-base-v2"
export RAY_RESULTS="ray_results_${MODEL}"
for DATA in "tweet_eval_hate" "tweet_eval_emotion"
do
  main ${DATA} ${MODEL}
  rm -rf ${RAY_RESULTS}
done