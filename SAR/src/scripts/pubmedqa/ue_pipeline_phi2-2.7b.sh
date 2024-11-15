
target_model_name="microsoft/phi-2" # OPT (13b, 125m, 250m, 1.3b, 30b), LLaMA (7b, 13b 30b?), GPT?
dataset_name="pub_med_qa"
num_of_gens=10
fraction_of_data=1
# for debug
# fraction_of_data=0.01
relevance_model_name="cross-encoder/stsb-roberta-large"
model_generation_temperature=0.5
max_length_of_generation=256
num_beams=1
devices=0,1,2,3

run_name=${target_model_name}/${dataset_name}/numbeams-${num_beams}/max_len_of_gen-${max_length_of_generation}

CUDA_VISIBLE_DEVICES=${devices} accelerate launch --multi_gpu generate.py --num-generations-per-prompt ${num_of_gens} --dataset ${dataset_name} \
--fraction-of-data-to-use ${fraction_of_data} --model ${target_model_name} --temperature ${model_generation_temperature} \
--run-name ${run_name} --max-length-of-generation ${max_length_of_generation} --num-beams ${num_beams}

CUDA_VISIBLE_DEVICES=${devices} python3 clean_generated_strings.py --generation-model ${target_model_name} --run-name ${run_name}

CUDA_VISIBLE_DEVICES=${devices} python3 get_semantic_clusters.py --generation-model ${target_model_name} --run-name ${run_name}

CUDA_VISIBLE_DEVICES=${devices} python3 get_likelihoods.py --evaluation-model ${target_model_name} --run-name ${run_name}

CUDA_VISIBLE_DEVICES=${devices} python3 get_tokenwise_importance.py --measurement-model ${relevance_model_name} \
--tokenizer-model ${target_model_name} --run-name ${run_name}

CUDA_VISIBLE_DEVICES=${devices} python3 get_sentence_similarities.py --measurement-model ${relevance_model_name} \
--run-name ${run_name}

CUDA_VISIBLE_DEVICES=${devices} python3 correctness_eval.py --run-name ${run_name}

CUDA_VISIBLE_DEVICES=${devices} python3 compute_uncertainty.py --senten-sim-meas-model ${relevance_model_name} \
--token-impt-meas-model ${relevance_model_name} --run-name ${run_name}
