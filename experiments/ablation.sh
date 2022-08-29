
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
  elif [ "${DATA}" = 'yelp_review' ]
    then
      MAX_LENGTH=256
  elif [ "${DATA}" = 'rct_sample' ]
    then
      MAX_LENGTH=256
  else
    MAX_LENGTH=64
  fi

  SPLIT='train'
  m3-rewrite -m "${MODEL}" -n 10 -k 10 -l ${MAX_LENGTH} -b ${BATCH} -d m3/multi_domain_document_classification --dataset-name "${DATA}" -s ${SPLIT} --dataset-column text -e "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10/${SPLIT}.json" -c ${CHUNK}

  # fine-tuning on vanilla dataset
  python lm_finetuning.py -m "${MODEL}" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.vanilla" --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-vanilla" --summary-file 'metric_summary.json'
  # fine-tuning on m3 dataset
  python lm_finetuning.py -m "${MODEL}" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v2" --summary-file 'metric_summary.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'train' --add-rewrite-text --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v2" --rewrite-dictionary-method 'best'
  python lm_finetuning.py -m "${MODEL}" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v3" --summary-file 'metric_summary.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'train' --add-rewrite-text --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --rewrite-dictionary-method 'largest_diff'
  python lm_finetuning.py -m "${MODEL}" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.replace-v2" --summary-file 'metric_summary.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'train' --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-replace-v2" --rewrite-dictionary-method 'best'
  python lm_finetuning.py -m "${MODEL}" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.replace-v3" --summary-file 'metric_summary.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'train' --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-replace-v3" --rewrite-dictionary-method 'largest_diff'

}


for DATA in "yelp_review" "chemprot" "citation_intent" "rct_sample" "sciie" "amcd" "tweet_eval_irony" "tweet_eval_hate" "tweet_eval_emotion"
do
  for MODEL in "albert-base-v2" "roberta-base"
  do
    main ${DATA} ${MODEL}
  done
done

MODEL="roberta-base"
for DATA in "yelp_review" "chemprot" "citation_intent" "rct_sample" "sciie" "amcd" "tweet_eval_irony" "tweet_eval_hate" "tweet_eval_emotion"
do
  main ${DATA} ${MODEL}
done

## Ablation
for SPLIT in 'validation' 'test'
do
  m3-rewrite -m ${MODEL} -n 10 -k 10 -l ${MAX_LENGTH} -b ${BATCH} -d m3/multi_domain_document_classification --dataset-name ${DATA} -s ${SPLIT} --dataset-column text -e "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10/${SPLIT}.json" -c ${CHUNK}
done
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


# evaluate m3 on m3 dataset fine-tuned model
python lm_finetuning.py -m "m3/m3-experiment-${MODEL}-${DATA//_/-}-add-v2" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v2/best_model" --skip-train --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v2" --summary-file 'metric_summary.edit.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'test'
python lm_finetuning.py -m "m3/m3-experiment-${MODEL}-${DATA//_/-}-replace-v2" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.replace-v2/best_model" --skip-train --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-replace-v2" --summary-file 'metric_summary.edit.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'test'
