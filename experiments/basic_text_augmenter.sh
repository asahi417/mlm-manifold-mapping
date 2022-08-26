
wandb offline
export WANDB_DISABLED='true'

"back_translation" "eda" "word_swapping_synonym" "word_swapping_embedding" "word_swapping_random"

main () {
  DATA=${1}
  TYPE=${2}
  SPLIT='train'
  m3-rewrite-basic-augmenter -d m3/multi_domain_document_classification --dataset-name "${DATA}" -s ${SPLIT} --dataset-column text -e "m3_output/m3_edit_inputs/basic_text_augmenter/${DATA}/${TYPE}/${SPLIT}.json" -a "${TYPE}" -t 100

  # fine-tuning on vanilla dataset
  python lm_finetuning.py -m "${MODEL}" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.vanilla" --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-vanilla" --summary-file 'metric_summary.json'
  # fine-tuning on m3 dataset
  python lm_finetuning.py -m "${MODEL}" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v2" --summary-file 'metric_summary.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'train' --add-rewrite-text --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v2"
  python lm_finetuning.py -m "${MODEL}" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v3" --summary-file 'metric_summary.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'train' --add-rewrite-text --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --rewrite-dictionary-method 'largest_diff'
  python lm_finetuning.py -m "${MODEL}" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.replace-v2" --summary-file 'metric_summary.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'train' --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-replace-v2"
  python lm_finetuning.py -m "${MODEL}" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.replace-v3" --summary-file 'metric_summary.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'train' --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-replace-v3" --rewrite-dictionary-method 'largest_diff'

}


for D in "yelp_review" "chemprot" "citation_intent" "rct_sample" "sciie" "amcd" "tweet_eval_irony" "tweet_eval_hate" "tweet_eval_emotion"
do
  for M in "albert-base-v2" "roberta-base"
  do
    main ${D} ${M}
  done
done