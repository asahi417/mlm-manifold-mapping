wandb offline
export WANDB_DISABLED='true'

main () {
  DATA=${1}
  TYPE=${2}
  SPLIT='train'
  m3-rewrite-basic-augmenter -d m3/multi_domain_document_classification --dataset-name "${DATA}" -s ${SPLIT} --dataset-column text -e "m3_output/m3_edit_inputs/basic_text_augmenter/${DATA}/${TYPE}/${SPLIT}.json" -a "${TYPE}" -t 100
}


for DATA in "yelp_review" "chemprot" "citation_intent" "rct_sample" "sciie" "amcd" "tweet_eval_irony" "tweet_eval_hate" "tweet_eval_emotion"
do
  for TYPE in "back_translation" "eda" "word_swapping_synonym" "word_swapping_embedding" "word_swapping_random"
  do
    main ${DATA} ${TYPE}
  done
done
