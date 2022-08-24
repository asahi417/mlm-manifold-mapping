wandb offline
export WANDB_DISABLED='true'

MODEL='albert-base-v2'

DATA='citation_intent'
MAX_LENGTH=128
python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v3" --summary-file 'metric_summary.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'train' --add-rewrite-text --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --rewrite-dictionary-method 'largest_diff'
python lm_finetuning.py -m "m3/m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v3/best_model" --skip-train --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --summary-file 'metric_summary.edit.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'test' --rewrite-dictionary-method 'largest_diff'

DATA='amcd'
MAX_LENGTH=64
python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v3" --summary-file 'metric_summary.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'train' --add-rewrite-text --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --rewrite-dictionary-method 'largest_diff'
python lm_finetuning.py -m "m3/m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v3/best_model" --skip-train --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --summary-file 'metric_summary.edit.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'test' --rewrite-dictionary-method 'largest_diff'

DATA='sciie'
MAX_LENGTH=128
python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v3" --summary-file 'metric_summary.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'train' --add-rewrite-text --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --rewrite-dictionary-method 'largest_diff'
python lm_finetuning.py -m "m3/m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v3/best_model" --skip-train --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --summary-file 'metric_summary.edit.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'test' --rewrite-dictionary-method 'largest_diff'

DATA='chemprot'
MAX_LENGTH=128
python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v3" --summary-file 'metric_summary.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'train' --add-rewrite-text --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --rewrite-dictionary-method 'largest_diff'
python lm_finetuning.py -m "m3/m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v3/best_model" --skip-train --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --summary-file 'metric_summary.edit.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'test' --rewrite-dictionary-method 'largest_diff'

DATA='tweet_eval_emotion'
MAX_LENGTH=64
python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v3" --summary-file 'metric_summary.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'train' --add-rewrite-text --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --rewrite-dictionary-method 'largest_diff'
python lm_finetuning.py -m "m3/m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v3/best_model" --skip-train --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --summary-file 'metric_summary.edit.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'test' --rewrite-dictionary-method 'largest_diff'

DATA='tweet_eval_hate'
MAX_LENGTH=64
python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v3" --summary-file 'metric_summary.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'train' --add-rewrite-text --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --rewrite-dictionary-method 'largest_diff'
python lm_finetuning.py -m "m3/m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v3/best_model" --skip-train --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --summary-file 'metric_summary.edit.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'test' --rewrite-dictionary-method 'largest_diff'

DATA='tweet_eval_irony'
MAX_LENGTH=64
python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v3" --summary-file 'metric_summary.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'train' --add-rewrite-text --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --rewrite-dictionary-method 'largest_diff'
python lm_finetuning.py -m "m3/m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v3/best_model" --skip-train --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --summary-file 'metric_summary.edit.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'test' --rewrite-dictionary-method 'largest_diff'

DATA='yelp_review'
MAX_LENGTH=256
python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v3" --summary-file 'metric_summary.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'train' --add-rewrite-text --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --rewrite-dictionary-method 'largest_diff'
python lm_finetuning.py -m "m3/m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v3/best_model" --skip-train --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --summary-file 'metric_summary.edit.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'test' --rewrite-dictionary-method 'largest_diff'

MODEL='roberta-base'

DATA='citation_intent'
MAX_LENGTH=128
python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v3" --summary-file 'metric_summary.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'train' --add-rewrite-text --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --rewrite-dictionary-method 'largest_diff'
python lm_finetuning.py -m "m3/m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v3/best_model" --skip-train --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --summary-file 'metric_summary.edit.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'test' --rewrite-dictionary-method 'largest_diff'

DATA='amcd'
MAX_LENGTH=64
python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v3" --summary-file 'metric_summary.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'train' --add-rewrite-text --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --rewrite-dictionary-method 'largest_diff'
python lm_finetuning.py -m "m3/m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v3/best_model" --skip-train --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --summary-file 'metric_summary.edit.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'test' --rewrite-dictionary-method 'largest_diff'

DATA='sciie'
MAX_LENGTH=128
python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v3" --summary-file 'metric_summary.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'train' --add-rewrite-text --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --rewrite-dictionary-method 'largest_diff'
python lm_finetuning.py -m "m3/m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v3/best_model" --skip-train --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --summary-file 'metric_summary.edit.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'test' --rewrite-dictionary-method 'largest_diff'

DATA='chemprot'
MAX_LENGTH=128
python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v3" --summary-file 'metric_summary.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'train' --add-rewrite-text --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --rewrite-dictionary-method 'largest_diff'
python lm_finetuning.py -m "m3/m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v3/best_model" --skip-train --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --summary-file 'metric_summary.edit.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'test' --rewrite-dictionary-method 'largest_diff'

DATA='tweet_eval_emotion'
MAX_LENGTH=64
python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v3" --summary-file 'metric_summary.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'train' --add-rewrite-text --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --rewrite-dictionary-method 'largest_diff'
python lm_finetuning.py -m "m3/m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v3/best_model" --skip-train --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --summary-file 'metric_summary.edit.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'test' --rewrite-dictionary-method 'largest_diff'

DATA='tweet_eval_hate'
MAX_LENGTH=64
python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v3" --summary-file 'metric_summary.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'train' --add-rewrite-text --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --rewrite-dictionary-method 'largest_diff'
python lm_finetuning.py -m "m3/m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v3/best_model" --skip-train --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --summary-file 'metric_summary.edit.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'test' --rewrite-dictionary-method 'largest_diff'

DATA='tweet_eval_irony'
MAX_LENGTH=64
python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v3" --summary-file 'metric_summary.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'train' --add-rewrite-text --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --rewrite-dictionary-method 'largest_diff'
python lm_finetuning.py -m "m3/m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v3/best_model" --skip-train --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --summary-file 'metric_summary.edit.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'test' --rewrite-dictionary-method 'largest_diff'

DATA='yelp_review'
MAX_LENGTH=256
python lm_finetuning.py -m ${MODEL} --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v3" --summary-file 'metric_summary.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'train' --add-rewrite-text --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --rewrite-dictionary-method 'largest_diff'
python lm_finetuning.py -m "m3/m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --dataset-name "${DATA}" -o "m3_output/ckpt/${MODEL}.${DATA}.add-v3/best_model" --skip-train --push-to-hub --hf-organization m3 -a "m3-experiment-${MODEL}-${DATA//_/-}-add-v3" --summary-file 'metric_summary.edit.json' --rewrite-dictionary-dir "m3_output/m3_edit_inputs/${MODEL}/${DATA}/length${MAX_LENGTH}.top10.iteration10" --rewrite-dictionary-split 'test' --rewrite-dictionary-method 'largest_diff'
