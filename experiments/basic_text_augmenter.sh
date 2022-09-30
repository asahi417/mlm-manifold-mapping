wandb offline
export WANDB_DISABLED='true'
export PARALLEL='0'
#
#export RAY_RESULTS='ray_results_1'
#DATA="amcd"
#TYPE='word_swapping_synonym'
#MODEL="albert-base-v2"
#SEED="4"
#python lm_finetuning.py -m "${MODEL}" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.${TYPE}.${SEED}" --summary-file 'metric_summary.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/basic_text_augmenter/${DATA}/${TYPE}" --rewrite-dictionary-split 'train' --add-rewrite-text --rewrite-dictionary-method ${SEED} \
#--push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-${TYPE//_/-}-${SEED}"
#rm -rf "${RAY_RESULTS}"
#rm -rf "m3_output/ckpt/${MODEL}.${DATA}.${TYPE}.${SEED}"
#



wandb offline
export WANDB_DISABLED='true'
export PARALLEL='0'
export RAY_RESULTS='ray_results'
for MODEL in 'roberta-base' 'albert-base-v2'
do
  for DATA in "amcd" "chemprot" "citation_intent" "rct_sample" "sciie" "tweet_eval_irony" "tweet_eval_hate" "tweet_eval_emotion"
  do

    for TYPE in "eda" "word_swapping_synonym" "word_swapping_embedding" "word_swapping_random"
    do
      for SEED in "0" "1" "2" "3" "4"
      do
        python lm_finetuning.py -m "${MODEL}" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.${TYPE}.${SEED}" --summary-file 'metric_summary.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/basic_text_augmenter/${DATA}/${TYPE}" --rewrite-dictionary-split 'train' --add-rewrite-text --rewrite-dictionary-method ${SEED} \
        --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-${TYPE//_/-}-${SEED}"
        rm -rf "${RAY_RESULTS}"
        rm -rf "m3_output/ckpt/${MODEL}.${DATA}.${TYPE}.${SEED}"
      done
    done

    TYPE="back_translation"
    python lm_finetuning.py -m "${MODEL}" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.${TYPE}" --summary-file 'metric_summary.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/basic_text_augmenter/${DATA}/${TYPE}" --rewrite-dictionary-split 'train' --add-rewrite-text \
    --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-${TYPE//_/-}"
    rm -rf "${RAY_RESULTS}"
    rm -rf "m3_output/ckpt/${MODEL}.${DATA}.${TYPE}"

  done
done