wandb offline
export WANDB_DISABLED='true'
#export PARALLEL='0'

NUM=10
DATA="citation_intent"
for DATA in "amcd" "chemprot" "citation_intent" # "rct_sample" "sciie" "tweet_eval_irony" "tweet_eval_hate" "tweet_eval_emotion"
#for DATA in "rct_sample" "sciie" "tweet_eval_irony" "tweet_eval_hate" "tweet_eval_emotion"
do
  for TYPE in "eda" "word_swapping_synonym" "word_swapping_embedding" "word_swapping_random"
  do
#    m3-rewrite-basic-augmenter -d m3/multi_domain_document_classification --dataset-name "${DATA}" -s "train" --dataset-column text -e "m3_output/m3_edit_inputs/basic_text_augmenter/${DATA}/${TYPE}/train.json" -a "${TYPE}" -t ${NUM}
#  done
    for MODEL in "albert-base-v2" "roberta-base"
    do
      for SEED in "0" "1" "2" "3" "4" "5" "6" "7" "8" "9"
      do
        python lm_finetuning.py -m "${MODEL}" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.${TYPE}.${SEED}" --summary-file 'metric_summary.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/basic_text_augmenter/${DATA}/${TYPE}" --rewrite-dictionary-split 'train' --add-rewrite-text --rewrite-dictionary-method ${SEED} \
        --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-${TYPE//_/-}-${SEED}"
        rm -rf ray_results
#        rm -rf "m3_output/ckpt/${MODEL}.${DATA}.${TYPE}.${SEED}"
      done
    done
  done

  TYPE="back_translation"
#  m3-rewrite-basic-augmenter -d m3/multi_domain_document_classification --dataset-name "${DATA}" -s "train" --dataset-column text -e "m3_output/m3_edit_inputs/basic_text_augmenter/${DATA}/${TYPE}/train.json" -a "${TYPE}" -t 1
  for MODEL in "albert-base-v2" "roberta-base"
    do
      python lm_finetuning.py -m "${MODEL}" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.${TYPE}.${SEED}" --summary-file 'metric_summary.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/basic_text_augmenter/${DATA}/${TYPE}" --rewrite-dictionary-split 'train' --add-rewrite-text --rewrite-dictionary-method "0" \
      --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-${TYPE//_/-}"
      rm -rf ray_results
#      rm -rf "m3_output/ckpt/${MODEL}.${DATA}.${TYPE}.${SEED}"
    done
done

#DATA='amcd'
#TYPE="back_translation"
#m3-rewrite-basic-augmenter -d m3/multi_domain_document_classification --dataset-name "${DATA}" -s "train" --dataset-column text -e "m3_output/m3_edit_inputs/basic_text_augmenter/${DATA}/${TYPE}/train.json" -a "${TYPE}" -t 1

