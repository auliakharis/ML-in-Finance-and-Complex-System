echo "<============================ New Model ===============================> " | tee -a output_llm/last_runs_9th_aug_after_21/10q/mixed_depth_10q.log
python -m script.benchmark.run_llm_eval_api_call_10q \
 --provider "Fireworks" --models "accounts/fireworks/models/qwen3p7-plus" --datasets 10q \
 --questions-10q script/compiler_pipeline_refactored_10Q/output/random_questions_10q_d1-5.csv \
 --output output_llm/last_runs_9th_aug_after_21/10q/qwen3p7-plus_mixed_depth.log  --limit 250 | tee -a output_llm/last_runs_9th_aug_after_21/10q/mixed_depth_10q.log

# ## echo "<============================ New Model ===============================> " | tee output_llm/10q/mixed_depth_10q.log
# ## python -m script.benchmark.run_llm_eval_api_call_10q \
# ##   --provider "Together" --models "Qwen/Qwen3.7-Plus" --datasets 10q \
# ##   --questions-10q script/compiler_pipeline_refactored_10Q/output/random_questions_10q_d1-5.csv \
# ##   --output output_llm/10q/qwen3-7_mixed_depth.log  --limit 250 | tee output_llm/10q/mixed_depth_10q.log
# #
# #echo "<============================ New Model ===============================> " | tee output_llm/10q/mixed_depth_10q.log
# #python -m script.benchmark.run_llm_eval_api_call_10q \
# #  --provider "Together" --models "Qwen/Qwen3.5-9B" --datasets 10q \
# #  --questions-10q script/compiler_pipeline_refactored_10Q/output/random_questions_10q_d1-5.csv \
# #  --output output_llm/10q/qwen3-5_mixed_depth.log  --limit 250 | tee output_llm/10q/mixed_depth_10q.log
# #
# #
# #echo "<============================ New Model ===============================> " | tee output_llm/10q/mixed_depth_10q.log
# #python -m script.benchmark.run_llm_eval_api_call_10q \
# #  --provider "Together" --models "meta-llama/Llama-3.3-70B-Instruct-Turbo" --datasets 10q \
# #  --questions-10q script/compiler_pipeline_refactored_10Q/output/random_questions_10q_d1-5.csv \
# #  --output output_llm/10q/llama_mixed_depth.log  --limit 250 | tee output_llm/10q/mixed_depth_10q.log
# #
# #echo "<============================ New Model ===============================> " | tee output_llm/10q/mixed_depth_10q.log
# #python -m script.benchmark.run_llm_eval_api_call_10q \
# #  --provider "Together" --models "openai/gpt-oss-120b" --datasets 10q \
# #  --questions-10q script/compiler_pipeline_refactored_10Q/output/random_questions_10q_d1-5.csv \
# #  --output output_llm/10q/gpt_oss_mixed_depth.log  --limit 250 | tee output_llm/10q/mixed_depth_10q.log
# #
# #echo "<============================ New Model ===============================> " | tee output_llm/10q/mixed_depth_10q.log
# #python -m script.benchmark.run_llm_eval_api_call_10q \
# #  --provider "Together" --models "google/gemma-4-31B-it" --datasets 10q \
# #  --questions-10q script/compiler_pipeline_refactored_10Q/output/random_questions_10q_d1-5.csv \
# #  --output output_llm/10q/gemma_mixed_depth.log  --limit 250 | tee output_llm/10q/mixed_depth_10q.log
# #
# #echo "<============================ New Model ===============================> " | tee output_llm/10q/mixed_depth_10q.log
# #python -m script.benchmark.run_llm_eval_api_call_10q \
# #  --provider "Together" --models "LiquidAI/LFM2.5-8B-A1B" --datasets 10q \
# #  --questions-10q script/compiler_pipeline_refactored_10Q/output/random_questions_10q_d1-5.csv \
# #  --output output_llm/10q/liquid_ai_mixed_depth.log  --limit 250 | tee output_llm/10q/mixed_depth_10q.log
# #
# #echo "<============================ New Model ===============================> " | tee output_llm/10q/mixed_depth_10q.log
# #python -m script.benchmark.run_llm_eval_api_call_10q \
# #  --provider "Apertus" --models "CSCS-Inference/swiss-ai/Apertus-70B-Instruct-2509" --datasets 10q \
# #  --questions-10q script/compiler_pipeline_refactored_10Q/output/random_questions_10q_d1-5.csv \
# #  --output output_llm/10q/Apertus-70B-Instruct-2509_mixed_depth.log  --limit 250 | tee output_llm/10q/mixed_depth_10q.log
# #
# #echo "<============================ New Model ===============================> " | tee output_llm/10q/mixed_depth_10q.log
# #python -m script.benchmark.run_llm_eval_api_call_10q \
# #  --provider "Apertus" --models "CSCS-Inference/swiss-ai/Apertus-v1.5-8B" --datasets 10q \
# #  --questions-10q script/compiler_pipeline_refactored_10Q/output/random_questions_10q_d1-5.csv \
# #  --output output_llm/10q/Apertus-v1.5-8B_mixed_depth.log  --limit 250 | tee output_llm/10q/mixed_depth_10q.log

# # echo "<============================ New Model ===============================> "   2>&1 |  tee output_llm/10q/depth_1_gemma.log
# # python -m script.benchmark.run_llm_eval_api_call_10q \
# #  --provider "Together" --models "google/gemma-4-31B-it" --datasets 10q \
# #  --questions-10q script/compiler_pipeline_refactored_10Q/output/random_questions_10q_d1.csv \
# #  --output output_llm/10q/gemma-4-31B-it-2509_depth_1.log  --limit 100   2>&1 |  tee output_llm/10q/depth_1_gemma.log

# # echo "<============================ New Model ===============================> "   2>&1 |  tee output_llm/10q/depth_1_deepseek.log
# # python -m script.benchmark.run_llm_eval_api_call_10q \
# #  --provider "Together" --models "deepseek-ai/DeepSeek-V4-Flash-0731" --datasets 10q \
# #  --questions-10q script/compiler_pipeline_refactored_10Q/output/random_questions_10q_d1.csv \
# #  --output output_llm/10q/DeepSeek-V4-Flash-0731_depth_1.log  --limit 100   2>&1 |  tee output_llm/10q/depth_1_deepseek.log

# # # <-->


# # echo "<============================ New Model ===============================> "   2>&1 |  tee output_llm/10q/depth_2_gemma.log
# # python -m script.benchmark.run_llm_eval_api_call_10q \
# #  --provider "Together" --models "google/gemma-4-31B-it" --datasets 10q \
# #  --questions-10q script/compiler_pipeline_refactored_10Q/output/random_questions_10q_d2.csv \
# #  --output output_llm/10q/gemma-4-31B-it-2509_depth_2.log  --limit 100   2>&1 |  tee output_llm/10q/depth_2_gemma.log

# # echo "<============================ New Model ===============================> "   2>&1 |  tee output_llm/10q/depth_2_deepseek.log
# # python -m script.benchmark.run_llm_eval_api_call_10q \
# #  --provider "Together" --models "deepseek-ai/DeepSeek-V4-Flash-0731" --datasets 10q \
# #  --questions-10q script/compiler_pipeline_refactored_10Q/output/random_questions_10q_d2.csv \
# #  --output output_llm/10q/DeepSeek-V4-Flash-0731_depth_2.log  --limit 100   2>&1 |  tee output_llm/10q/depth_2_deepseek.log

# # # <-->

# # echo "<============================ New Model ===============================> "   2>&1 |  tee output_llm/10q/depth_3_gemma.log
# # python -m script.benchmark.run_llm_eval_api_call_10q \
# #  --provider "Together" --models "google/gemma-4-31B-it" --datasets 10q \
# #  --questions-10q script/compiler_pipeline_refactored_10Q/output/random_questions_10q_d3.csv \
# #  --output output_llm/10q/gemma-4-31B-it-2509_depth_3.log  --limit 100   2>&1 |  tee output_llm/10q/depth_3_gemma.log

# # echo "<============================ New Model ===============================> "   2>&1 |  tee output_llm/10q/depth_3_deepseek.log
# # python -m script.benchmark.run_llm_eval_api_call_10q \
# #  --provider "Together" --models "deepseek-ai/DeepSeek-V4-Flash-0731" --datasets 10q \
# #  --questions-10q script/compiler_pipeline_refactored_10Q/output/random_questions_10q_d3.csv \
# #  --output output_llm/10q/DeepSeek-V4-Flash-0731_depth_3.log  --limit 100   2>&1 |  tee output_llm/10q/depth_3_deepseek.log

# # # <-->

# # echo "<============================ New Model ===============================> "   2>&1 |  tee output_llm/10q/depth_4_gemma.log
# # python -m script.benchmark.run_llm_eval_api_call_10q \
# #  --provider "Together" --models "google/gemma-4-31B-it" --datasets 10q \
# #  --questions-10q script/compiler_pipeline_refactored_10Q/output/random_questions_10q_d4.csv \
# #  --output output_llm/10q/gemma-4-31B-it-2509_depth_4.log  --limit 100   2>&1 |  tee output_llm/10q/depth_4_gemma.log

# # echo "<============================ New Model ===============================> "   2>&1 |  tee output_llm/10q/depth_4_deepseek.log
# # python -m script.benchmark.run_llm_eval_api_call_10q \
# #  --provider "Together" --models "deepseek-ai/DeepSeek-V4-Flash-0731" --datasets 10q \
# #  --questions-10q script/compiler_pipeline_refactored_10Q/output/random_questions_10q_d4.csv \
# #  --output output_llm/10q/DeepSeek-V4-Flash-0731_depth_4.log  --limit 100   2>&1 |  tee output_llm/10q/depth_4_deepseek.log

# # # <-->

# # echo "<============================ New Model ===============================> "   2>&1 |  tee output_llm/10q/missing_values_deepseek.log
# # python -m script.benchmark.run_llm_eval_api_call_10q \
# #  --provider "Together" --models "deepseek-ai/DeepSeek-V4-Flash-0731" --datasets 10q_missing \
# #  --questions-10q script/compiler_pipeline_refactored_10Q/output/random_questions_10q_d1-5.csv \
# #  --output output_llm/10q/DeepSeek-V4-Flash-0731_missing.log  --limit 100   2>&1 |  tee output_llm/10q/missing_values_deepseek.log

# # echo "<============================ New Model ===============================> "   2>&1 |  tee output_llm/10q/garbage_deepseek.log
# # python -m script.benchmark.run_llm_eval_api_call_10q \
# #  --provider "Together" --models "deepseek-ai/DeepSeek-V4-Flash-0731" --datasets 10q_garbage \
# #  --questions-10q script/compiler_pipeline_refactored_10Q/output/random_questions_10q_d1-5.csv \
# #  --output output_llm/10q/DeepSeek-V4-Flash-0731_garbage.log  --limit 100   2>&1 |  tee output_llm/10q/garbage_deepseek.log

# # echo "<============================ New Model ===============================> "   2>&1 |  tee output_llm/10q/look_alike_deepseek.log
# # python -m script.benchmark.run_llm_eval_api_call_10q \
# #  --provider "Together" --models "deepseek-ai/DeepSeek-V4-Flash-0731" --datasets 10q_lookalike \
# #  --questions-10q script/compiler_pipeline_refactored_10Q/output/random_questions_10q_d1-5.csv \
# #  --output output_llm/10q/DeepSeek-V4-Flash-0731_lookalike.log  --limit 100   2>&1 |  tee output_llm/10q/look_alike_deepseek.log

# # echo "<============================ New Model ===============================> "   2>&1 |  tee output_llm/10q/scaling_deepseek.log
# # python -m script.benchmark.run_llm_eval_api_call_10q \
# #  --provider "Together" --models "deepseek-ai/DeepSeek-V4-Flash-0731" --datasets 10q_scaling \
# #  --questions-10q script/compiler_pipeline_refactored_10Q/output/random_questions_10q_d1-5.csv \
# #  --output output_llm/10q/DeepSeek-V4-Flash-0731_scaling.log  --limit 100   2>&1 |  tee output_llm/10q/scaling_deepseek.log

# echo "<============================ New Model ===============================> "   2>&1 |  tee output_llm/10q/useless_info_deepseek.log
# python -m script.benchmark.run_llm_eval_api_call_10q \
#  --provider "Together" --models "deepseek-ai/DeepSeek-V4-Flash-0731" --datasets 10q_useless_info \
#  --questions-10q script/compiler_pipeline_refactored_10Q/output/random_questions_10q_d1-5.csv \
#  --output output_llm/10q/DeepSeek-V4-Flash-0731_useless_info.log  --limit 100   2>&1 |  tee output_llm/10q/useless_info_deepseek.log

# # echo "<============================ New Model ===============================> "   2>&1 |  tee output_llm/10q/cross_sheet_contamnation_deepseek.log
# # python -m script.benchmark.run_llm_eval_api_call_10q \
# #  --provider "Together" --models "deepseek-ai/DeepSeek-V4-Flash-0731" --datasets 10q_cross \
# #  --questions-10q script/compiler_pipeline_refactored_10Q/output/random_questions_10q_d1-5.csv \
# #  --output output_llm/10q/DeepSeek-V4-Flash-0731_cross.log  --limit 100   2>&1 |  tee output_llm/10q/cross_sheet_contamnation_deepseek.log

# echo "<============================ New Model ===============================> "   2>&1 |  tee output_llm/10q/combined_deepseek.log
# python -m script.benchmark.run_llm_eval_api_call_10q \
#  --provider "Together" --models "deepseek-ai/DeepSeek-V4-Flash-0731" --datasets 10q_combined_all \
#  --questions-10q script/compiler_pipeline_refactored_10Q/output/random_questions_10q_d1-5.csv \
#  --output output_llm/10q/DeepSeek-V4-Flash-0731_combined_all.log  --limit 100   2>&1 |  tee output_llm/10q/combined_deepseek.log





# # echo "<============================ New Model ===============================> "   2>&1 |  tee output_llm/10q/missing_values_gemma.log
# # python -m script.benchmark.run_llm_eval_api_call_10q \
# #  --provider "Together" --models "google/gemma-4-31B-it" --datasets 10q_missing \
# #  --questions-10q script/compiler_pipeline_refactored_10Q/output/random_questions_10q_d1-5.csv \
# #  --output output_llm/10q/gemma-4-31B-it_missing.log  --limit 100   2>&1 |  tee output_llm/10q/missing_values_gemma.log

# # echo "<============================ New Model ===============================> "   2>&1 |  tee output_llm/10q/garbage_gemma.log
# # python -m script.benchmark.run_llm_eval_api_call_10q \
# #  --provider "Together" --models "google/gemma-4-31B-it" --datasets 10q_garbage \
# #  --questions-10q script/compiler_pipeline_refactored_10Q/output/random_questions_10q_d1-5.csv \
# #  --output output_llm/10q/gemma-4-31B-it_garbage.log  --limit 100   2>&1 |  tee output_llm/10q/garbage_gemma.log

# # echo "<============================ New Model ===============================> "   2>&1 |  tee output_llm/10q/look_alike_gemma.log
# # python -m script.benchmark.run_llm_eval_api_call_10q \
# #  --provider "Together" --models "google/gemma-4-31B-it" --datasets 10q_lookalike \
# #  --questions-10q script/compiler_pipeline_refactored_10Q/output/random_questions_10q_d1-5.csv \
# #  --output output_llm/10q/gemma-4-31B-it_lookalike.log  --limit 100   2>&1 |  tee output_llm/10q/look_alike_gemma.log

# # echo "<============================ New Model ===============================> "   2>&1 |  tee output_llm/10q/scaling_gemma.log
# # python -m script.benchmark.run_llm_eval_api_call_10q \
# #  --provider "Together" --models "google/gemma-4-31B-it" --datasets 10q_scaling \
# #  --questions-10q script/compiler_pipeline_refactored_10Q/output/random_questions_10q_d1-5.csv \
# #  --output output_llm/10q/gemma-4-31B-it_scaling.log  --limit 100   2>&1 |  tee output_llm/10q/scaling_gemma.log

# echo "<============================ New Model ===============================> "   2>&1 |  tee output_llm/10q/useless_info_gemma.log
# python -m script.benchmark.run_llm_eval_api_call_10q \
#  --provider "Together" --models "google/gemma-4-31B-it" --datasets 10q_useless_info \
#  --questions-10q script/compiler_pipeline_refactored_10Q/output/random_questions_10q_d1-5.csv \
#  --output output_llm/10q/gemma-4-31B-it_useless_info.log  --limit 100   2>&1 |  tee output_llm/10q/useless_info_gemma.log

# # echo "<============================ New Model ===============================> "   2>&1 |  tee output_llm/10q/cross_sheet_contamnation_gemma.log
# # python -m script.benchmark.run_llm_eval_api_call_10q \
# #  --provider "Together" --models "google/gemma-4-31B-it" --datasets 10q_cross \
# #  --questions-10q script/compiler_pipeline_refactored_10Q/output/random_questions_10q_d1-5.csv \
# #  --output output_llm/10q/gemma-4-31B-it_cross.log  --limit 100   2>&1 |  tee output_llm/10q/cross_sheet_contamnation_gemma.log

# echo "<============================ New Model ===============================> "   2>&1 |  tee output_llm/10q/combined_gemma.log
# python -m script.benchmark.run_llm_eval_api_call_10q \
#  --provider "Together" --models "google/gemma-4-31B-it" --datasets 10q_combined_all \
#  --questions-10q script/compiler_pipeline_refactored_10Q/output/random_questions_10q_d1-5.csv \
#  --output output_llm/10q/gemma-4-31B-it_combined_all.log  --limit 100   2>&1 |  tee output_llm/10q/combined_gemma.log
