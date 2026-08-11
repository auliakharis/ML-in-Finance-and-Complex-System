# #python script/compiler_pipeline_adversarial/data_prep_adversarial.py
# #python script/compiler_pipeline_adversarial/make_random_questions_adversarial.py --n 100 --depth-min 3 --depth-max 3
# #echo "<===== new run accounts/fireworks/models/qwen3p7-plus depth_3 =======>" | tee -a depth_3.log
# #python -u script/benchmark/run_llm_eval_api_call.py --provider "Fireworks" --models "accounts/fireworks/models/qwen3p7-plus" --datasets 90q --limit 100 --output output_llm/3p7-plu_depth_3.json |& tee -a depth_3.log
# #echo "<===== new run Qwen/Qwen3.5-9B depth_3=======>" | tee -a depth_3.log
# #python -u script/benchmark/run_llm_eval_api_call.py --provider "Together" --models "Qwen/Qwen3.5-9B" --datasets 90q --limit 100 --output output_llm/Qwen3.5-9B_depth_3.json |& tee -a depth_3.log
# #
# #python script/compiler_pipeline_adversarial/make_random_questions_adversarial.py --n 100 --depth-min 4 --depth-max 4
# #echo "<===== new run accounts/fireworks/models/qwen3p7-plus depth_4=======>" | tee -a depth_4.log
# #python -u script/benchmark/run_llm_eval_api_call.py --provider "Fireworks" --models "accounts/fireworks/models/qwen3p7-plus" --datasets 90q --limit 100 --output output_llm/3p7-plu_depth_4.json |& tee -a depth_4.log
# #echo "<===== new run Qwen/Qwen3.5-9B depth_4=======>" | tee -a depth_4.log
# #python -u script/benchmark/run_llm_eval_api_call.py --provider "Together" --models "Qwen/Qwen3.5-9B" --datasets 90q --limit 100 --output output_llm/Qwen3.5-depth_4.json |& tee -a depth_4.log
# #
# #python script/compiler_pipeline_adversarial/make_random_questions_adversarial.py --n 100 --depth-min 5 --depth-max 5
# #echo "<===== new run accounts/fireworks/models/qwen3p7-plus depth_5=======>" | tee -a depth_5.log
# #python -u script/benchmark/run_llm_eval_api_call.py --provider "Fireworks" --models "accounts/fireworks/models/qwen3p7-plus" --datasets 90q --limit 100 --output output_llm/3p7-plu_depth_5.json |& tee -a depth_5.log
# #echo "<===== new run Qwen/Qwen3.5-9B depth_5=======>" | tee -a depth_5.log
# #python -u script/benchmark/run_llm_eval_api_call.py --provider "Together" --models "Qwen/Qwen3.5-9B" --datasets 90q --limit 100 --output output_llm/Qwen3.5-9B_depth_5.json |& tee -a depth_5.log
# #
# #python script/compiler_pipeline_adversarial/make_random_questions_adversarial.py --n 100 --depth-min 6 --depth-max 6
# #echo "<===== new run accounts/fireworks/models/qwen3p7-plus depth_6=======>" | tee -a depth_6.log
# #python -u script/benchmark/run_llm_eval_api_call.py --provider "Fireworks" --models "accounts/fireworks/models/qwen3p7-plus" --datasets 90q --limit 100 --output output_llm/3p7-plu_depth_6.json |& tee -a depth_6.log
# #echo "<===== new run Qwen/Qwen3.5-9B depth_6=======>" | tee -a depth_6.log
# #python -u script/benchmark/run_llm_eval_api_call.py --provider "Together" --models "Qwen/Qwen3.5-9B" --datasets 90q --limit 100 --output output_llm/Qwen3.5-9B_depth_6.json |& tee -a depth_6.log
# #
# #python script/compiler_pipeline_adversarial/make_random_questions_adversarial.py --n 100 --depth-min 7 --depth-max 7
# #echo "<===== new run accounts/fireworks/models/qwen3p7-plus depth_7=======>" | tee -a depth_7.log
# #python -u script/benchmark/run_llm_eval_api_call.py --provider "Fireworks" --models "accounts/fireworks/models/qwen3p7-plus" --datasets 90q --limit 100 --output output_llm/3p7-plu_ddepth_7.json |& tee -a depth_7.log
# #echo "<===== new run Qwen/Qwen3.5-9B depth_7=======>" | tee -a depth_7.log
# #python -u script/benchmark/run_llm_eval_api_call.py --provider "Together" --models "Qwen/Qwen3.5-9B" --datasets 90q --limit 100 --output output_llm/Qwen3.5-9B_depth_7.json |& tee -a depth_7.log
# #
# #python script/compiler_pipeline_adversarial/make_random_questions_adversarial.py --n 100 --depth-min 8  --depth-max 8
# #echo "<===== new run accounts/fireworks/models/qwen3p7-plus depth_8=======>" | tee -a depth_8.log
# #python -u script/benchmark/run_llm_eval_api_call.py --provider "Fireworks" --models "accounts/fireworks/models/qwen3p7-plus" --datasets 90q --limit 100 --output output_llm/3p7-plu_depth_8.json |& tee -a depth_8.log
# #echo "<===== new run Qwen/Qwen3.5-9B depth_8=======>" | tee -a depth_8.log
# #python -u script/benchmark/run_llm_eval_api_call.py --provider "Together" --models "Qwen/Qwen3.5-9B" --datasets 90q --limit 100 --output output_llm/Qwen3.5-9B_depth_8.json |& tee -a depth_8.log
# #
# #python script/compiler_pipeline_adversarial/make_random_questions_adversarial.py --n 100 --depth-min 0 --depth-max 0 --derived-prob-min 0 --derived-prob-max 0
# #python -u script/benchmark/run_llm_eval_api_call.py --provider "Apertus" --models "accounts/fireworks/models/deepseek-v4-flash" --datasets 90q --limit 100 --output output_llm/Deepseek_test_depth_0.json |& tee -a Deepseek_test_depth_0.log
# #
# #python script/compiler_pipeline_adversarial/make_random_questions_adversarial.py --n 100 --depth-min 1 --depth-max 1 --derived-prob-min 0 --derived-prob-max 0
# #python -u script/benchmark/run_llm_eval_api_call.py --provider "Fireworks" --models "accounts/fireworks/models/deepseek-v4-flash" --datasets 90q --limit 100 --output output_llm/Deepseek_test_depth_1.json |& tee -a Deepseek_test_depth_1.log
# #
# #python script/compiler_pipeline_adversarial/make_random_questions_adversarial.py --n 100 --depth-min 2 --depth-max 2 --derived-prob-min 0 --derived-prob-max 0
# #python -u script/benchmark/run_llm_eval_api_call.py --provider "Fireworks" --models "accounts/fireworks/models/deepseek-v4-flash" --datasets 90q --limit 100 --output output_llm/Deepseek_test_depth_2.json |& tee -a Deepseek_test_depth_2.log
# #
# #python script/compiler_pipeline_adversarial/make_random_questions_adversarial.py --n 100 --depth-min 3 --depth-max 3
# #python -u script/benchmark/run_llm_eval_api_call.py --provider "Fireworks" --models "accounts/fireworks/models/deepseek-v4-flash" --datasets 90q --limit 100 --output output_llm/Deepseek_test_depth_3.json |& tee -a Deepseek_test_depth_3.log
# #
# #python script/compiler_pipeline_adversarial/make_random_questions_adversarial.py --n 100 --depth-min 4 --depth-max 4
# #python -u script/benchmark/run_llm_eval_api_call.py --provider "Fireworks" --models "accounts/fireworks/models/deepseek-v4-flash" --datasets 90q --limit 100 --output output_llm/Deepseek_test_depth_4.json |& tee -a Deepseek_test_depth_4.log
# #
# #python script/compiler_pipeline_adversarial/make_random_questions_adversarial.py --n 100 --depth-min 5 --depth-max 5
# #python -u script/benchmark/run_llm_eval_api_call.py --provider "Fireworks" --models "accounts/fireworks/models/deepseek-v4-flash" --datasets 90q --limit 100 --output output_llm/Deepseek_test_depth_5.json |& tee -a Deepseek_test_depth_5.log
# #
# #python script/compiler_pipeline_adversarial/make_random_questions_adversarial.py --n 100 --depth-min 6 --depth-max 6
# #python -u script/benchmark/run_llm_eval_api_call.py --provider "Fireworks" --models "accounts/fireworks/models/deepseek-v4-flash" --datasets 90q --limit 100 --output output_llm/Deepseek_test_depth_6.json |& tee -a Deepseek_test_depth_6.log
# #
# #python script/compiler_pipeline_adversarial/make_random_questions_adversarial.py --n 100 --depth-min 7 --depth-max 7
# #python -u script/benchmark/run_llm_eval_api_call.py --provider "Fireworks" --models "accounts/fireworks/models/deepseek-v4-flash" --datasets 90q --limit 100 --output output_llm/Deepseek_test_depth_7.json |& tee -a Deepseek_test_depth_7.log
# #
# #python script/compiler_pipeline_adversarial/make_random_questions_adversarial.py --n 100 --depth-min 8 --depth-max 8
# #python -u script/benchmark/run_llm_eval_api_call.py --provider "Fireworks" --models "accounts/fireworks/models/deepseek-v4-flash" --datasets 90q --limit 100 --output output_llm/Deepseek_test_depth_8.json |& tee -a Deepseek_test_depth_8.log


# # Base depth 1-5;
# #python script/compiler_pipeline_adversarial/make_random_questions_adversarial.py --n 200 --depth-min 1 --depth-max 5
# #python script/compiler_pipeline_adversarial/adversarial.py

# # missing_values
# #cp script/compiler_pipeline_adversarial/output/adversarial/missing_values.json script/compiler_pipeline_adversarial/output/financial_spreadsheet.json
# #python script/benchmark/run_llm_eval_api_call.py --models swiss-ai/Apertus-v1.5-70B --datasets 90q --limit 10
# #python script/benchmark/run_llm_eval_api_call.py --models swiss-ai/Apertus-v1.5-70B --datasets 90q --limit 100
# #
# ## garbage
# #cp script/compiler_pipeline_adversarial/output/adversarial/garbage.json script/compiler_pipeline_adversarial/output/financial_spreadsheet.json
# #python script/benchmark/run_llm_eval_api_call.py --provider "Apertus"  --models "swiss-ai/Apertus-v1.5-70B" --datasets 90q --limit 100
# #python script/benchmark/run_llm_eval_api_call.py --models swiss-ai/Apertus-v1.5-70B --datasets 90q --limit 100
# #
# ## lookalike
# #cp script/compiler_pipeline_adversarial/output/adversarial/lookalike.json script/compiler_pipeline_adversarial/output/financial_spreadsheet.json
# #python script/benchmark/run_llm_eval_api_call.py --models swiss-ai/Apertus-v1.5-70B --datasets 90q --limit 100
# ##python script/benchmark/run_llm_eval_api_call.py --models swiss-ai/Apertus-v1.5-70B --datasets 90q --limit 100
# ##
# ### cross_contaminated
# #cp script/compiler_pipeline_adversarial/output/adversarial/cross_contaminated.json script/compiler_pipeline_adversarial/output/financial_spreadsheet.json
# #python script/benchmark/run_llm_eval_api_call.py --models swiss-ai/Apertus-v1.5-70B --datasets 90q --limit 100
# ##python script/benchmark/run_llm_eval_api_call.py --models swiss-ai/Apertus-v1.5-70B --datasets 90q --limit 100
# ##
# ## all together
# #cp script/compiler_pipeline_adversarial/output/adversarial/combined_adversarial.json script/compiler_pipeline_adversarial/output/financial_spreadsheet.json
# # python script/benchmark/run_llm_eval_api_call.py --provider "Apertus" --models swiss-ai/Apertus-v1.5-70B --datasets 90q --limit 100
# # python script/benchmark/run_llm_eval_api_call.py --models swiss-ai/Apertus-v1.5-70B --datasets 90q --limit 100

# #python script/compiler_pipeline_adversarial/make_random_questions_adversarial.py --n 100 --depth-min 8 --depth-max 8
# python -m script.benchmark.run_llm_eval_api_call --provider "Fireworks" --models "accounts/fireworks/models/deepseek-v4-flash" --limit 10 --datasets multiturn-90q

# python script/compiler_pipeline_adversarial/data_prep_adversarial.py
# # python script/compiler_pipeline_adversarial/make_random_questions_adversarial.py --n 250 --depth-min 1 --depth-max 5
# # echo "<===== new run accounts/fireworks/models/qwen3p7-plus depth_3 =======>" | tee -a mixed_depth_supplimental.log
# # python -u script/benchmark/run_llm_eval_api_call.py --provider "Fireworks" --models "accounts/fireworks/models/qwen3p7-plus" --datasets 90q --limit 100 --output output_llm/3p7-plu_depth_3.json |& tee -a depth_3.log

# python script/compiler_pipeline_adversarial/data_prep_adversarial.py
# python script/compiler_pipeline_adversarial/make_random_questions_adversarial.py --n 250 --depth-min 1 --depth-max 5
# echo "<===== new run accounts/fireworks/models/qwen3p7-plus depth_1-5 =======>" 2>&1 |  tee -a output_llm/last_runs_9th_aug_after_21/mixed_depth_qwen3p7plus.log
# python -m script.benchmark.run_llm_eval_api_call --provider "Fireworks" --models "accounts/fireworks/models/qwen3p7-plus" --datasets 90q --limit 250 --output output_llm/last_runs_9th_aug_after_21/mixed_depth_qwen3p7plus.json 2>&1 |   tee -a output_llm/last_runs_9th_aug_after_21/mixed_depth_qwen3p7plus.log

# echo "<===== new run CSCS-Inference/swiss-ai/Apertus-v1.5-70B depth_1-5 =======>" 2>&1 |  tee -a output_llm/last_runs_9th_aug_after_21/mixed_depth_apertus_70b.log
# python -m script.benchmark.run_llm_eval_api_call --provider "Apertus" --models "CSCS-Inference/swiss-ai/Apertus-v1.5-70B" --datasets 90q --limit 10 --output output_llm/last_runs_9th_aug_after_21/apertus_70b.json 2>&1 |   tee -a output_llm/last_runs_9th_aug_after_21/mixed_depth_apertus_70b.log

# echo "<===== new run CSCS-Inference/swiss-ai/Apertus-v1.5-8B depth_1-5 =======>" 2>&1 |  tee -a output_llm/last_runs_9th_aug_after_21/mixed_depth_apertus_8b.log
# python -m script.benchmark.run_llm_eval_api_call --provider "Apertus" --models "CSCS-Inference/swiss-ai/Apertus-v1.5-8B" --datasets 90q --limit 10 --output output_llm/last_runs_9th_aug_after_21/apertus_8b.json 2>&1 |   tee -a output_llm/last_runs_9th_aug_after_21/mixed_depth_apertus_8b.log