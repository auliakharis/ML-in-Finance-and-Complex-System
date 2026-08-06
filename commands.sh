# < ---------------->
# Base depth 1-5
python script/compiler_pipeline_adversarial/make_random_questions_adversarial.py --n 100 --depth-min 1 --depth-max 5
python script/benchmark/run_llm_eval_api_call.py --models swiss-ai/Apertus-v1.5-70B --datasets 90q --limit 100 --output output_llm/example_special_name.json
python script/benchmark/run_llm_eval_api_call.py --models swiss-ai/Apertus-v1.5-70B --datasets 90q --limit 100 |& tee -a llm_eval.log
echo "<===== new run =======>" | tee -a output.log
python script/benchmark/run_llm_eval_api_call.py --models swiss-ai/Apertus-v1.5-70B --datasets 90q --limit 100 |& tee -a llm_eval.log
echo "<===== new run =======>" | tee -a output.log
python script/benchmark/run_llm_eval_api_call.py --models swiss-ai/Apertus-v1.5-70B --datasets 90q --limit 100 |& tee -a llm_eval.log
echo "<===== new run =======>" | tee -a output.log
python script/benchmark/run_llm_eval_api_call.py --models swiss-ai/Apertus-v1.5-70B --datasets 90q --limit 100 |& tee -a llm_eval.log
echo "<===== new run =======>" | tee -a output.log
python script/benchmark/run_llm_eval_api_call.py --models swiss-ai/Apertus-v1.5-70B --datasets 90q --limit 100 |& tee -a llm_eval.log
echo "<===== new run =======>" | tee -a output.log
python script/benchmark/run_llm_eval_api_call.py --models swiss-ai/Apertus-v1.5-70B --datasets 90q --limit 100 |& tee -a llm_eval.log
echo "<===== new run =======>" | tee -a output.log
python script/benchmark/run_llm_eval_api_call.py --models swiss-ai/Apertus-v1.5-70B --datasets 90q --limit 100 |& tee -a llm_eval.log
echo "<===== new run =======>" | tee -a output.log
python script/benchmark/run_llm_eval_api_call.py --models swiss-ai/Apertus-v1.5-70B --datasets 90q --limit 100 |& tee -a llm_eval.log

# < ---------------->
# Base depth 1
python script/compiler_pipeline_adversarial/make_random_questions_adversarial.py --n 100 --depth-min 1 --depth-max 1
echo "<===== new run =======>" | tee -a output.log
python script/benchmark/run_llm_eval_api_call.py --models swiss-ai/Apertus-v1.5-70B --datasets 90q --limit 100 |& tee -a llm_eval.log
echo "<===== new run =======>" | tee -a output.log
python script/benchmark/run_llm_eval_api_call.py --models swiss-ai/Apertus-v1.5-70B --datasets 90q --limit 100 |& tee -a llm_eval.log

# < ---------------->
# Base depth 2
python script/compiler_pipeline_adversarial/make_random_questions_adversarial.py --n 100 --depth-min 2 --depth-max 2
python script/benchmark/run_llm_eval_api_call.py --models swiss-ai/Apertus-v1.5-70B --datasets 90q --limit 100
python script/benchmark/run_llm_eval_api_call.py --models swiss-ai/Apertus-v1.5-70B --datasets 90q --limit 100

# < ---------------->
# Base depth 3
python script/compiler_pipeline_adversarial/make_random_questions_adversarial.py --n 100 --depth-min 3 --depth-max 3
python script/benchmark/run_llm_eval_api_call.py --models swiss-ai/Apertus-v1.5-70B --datasets 90q --limit 100
python script/benchmark/run_llm_eval_api_call.py --models swiss-ai/Apertus-v1.5-70B --datasets 90q --limit 100

# < ---------------->
# Base depth 4
python script/compiler_pipeline_adversarial/make_random_questions_adversarial.py --n 100 --depth-min 4 --depth-max 4
python script/benchmark/run_llm_eval_api_call.py --models swiss-ai/Apertus-v1.5-70B --datasets 90q --limit 100
python script/benchmark/run_llm_eval_api_call.py --models swiss-ai/Apertus-v1.5-70B --datasets 90q --limit 100

# < ---------------->
# Base depth 5
python script/compiler_pipeline_adversarial/make_random_questions_adversarial.py --n 100 --depth-min 5 --depth-max 5
python script/benchmark/run_llm_eval_api_call.py --models swiss-ai/Apertus-v1.5-70B --datasets 90q --limit 100
python script/benchmark/run_llm_eval_api_call.py --models swiss-ai/Apertus-v1.5-70B --datasets 90q --limit 100

# < ---------------->
# Base depth 6
python script/compiler_pipeline_adversarial/make_random_questions_adversarial.py --n 100 --depth-min 6 --depth-max 6
python script/benchmark/run_llm_eval_api_call.py --models swiss-ai/Apertus-v1.5-70B --datasets 90q --limit 100
python script/benchmark/run_llm_eval_api_call.py --models swiss-ai/Apertus-v1.5-70B --datasets 90q --limit 100

# < ---------------->
# Base depth 7
python script/compiler_pipeline_adversarial/make_random_questions_adversarial.py --n 100 --depth-min 7 --depth-max 7
python script/benchmark/run_llm_eval_api_call.py --models swiss-ai/Apertus-v1.5-70B --datasets 90q --limit 100
python script/benchmark/run_llm_eval_api_call.py --models swiss-ai/Apertus-v1.5-70B --datasets 90q --limit 100

# < ---------------->
# Base depth 8
python script/compiler_pipeline_adversarial/make_random_questions_adversarial.py --n 100 --depth-min 8 --depth-max 8
python script/benchmark/run_llm_eval_api_call.py --models swiss-ai/Apertus-v1.5-70B --datasets 90q --limit 100
python script/benchmark/run_llm_eval_api_call.py --models swiss-ai/Apertus-v1.5-70B --datasets 90q --limit 100

# < ---------------->
# Base depth 1-5;
python script/compiler_pipeline_adversarial/make_random_questions_adversarial.py --n 100 --depth-min 1 --depth-max 5
python script/compiler_pipeline_adversarial/adversarial.py

# missing_values
cp script/compiler_pipeline_adversarial/output/adversarial/missing_values.csv script/compiler_pipeline_adversarial/output/financial_spreadsheet.csv
python script/benchmark/run_llm_eval_api_call.py --models swiss-ai/Apertus-v1.5-70B --datasets 90q --limit 100
python script/benchmark/run_llm_eval_api_call.py --models swiss-ai/Apertus-v1.5-70B --datasets 90q --limit 100

# garbage
cp script/compiler_pipeline_adversarial/output/adversarial/garbage.csv script/compiler_pipeline_adversarial/output/financial_spreadsheet.csv
python script/benchmark/run_llm_eval_api_call.py --models swiss-ai/Apertus-v1.5-70B --datasets 90q --limit 100
python script/benchmark/run_llm_eval_api_call.py --models swiss-ai/Apertus-v1.5-70B --datasets 90q --limit 100

# lookalike
cp script/compiler_pipeline_adversarial/output/adversarial/lookalike.csv script/compiler_pipeline_adversarial/output/financial_spreadsheet.csv
python script/benchmark/run_llm_eval_api_call.py --models swiss-ai/Apertus-v1.5-70B --datasets 90q --limit 100
python script/benchmark/run_llm_eval_api_call.py --models swiss-ai/Apertus-v1.5-70B --datasets 90q --limit 100

# cross_contaminated
cp script/compiler_pipeline_adversarial/output/adversarial/cross_contaminated.csv script/compiler_pipeline_adversarial/output/financial_spreadsheet.csv
python script/benchmark/run_llm_eval_api_call.py --models swiss-ai/Apertus-v1.5-70B --datasets 90q --limit 100
python script/benchmark/run_llm_eval_api_call.py --models swiss-ai/Apertus-v1.5-70B --datasets 90q --limit 100

# all together
cp script/compiler_pipeline_adversarial/output/adversarial/combined_adversarial.csv script/compiler_pipeline_adversarial/output/financial_spreadsheet.csv
python script/benchmark/run_llm_eval_api_call.py --models swiss-ai/Apertus-v1.5-70B --datasets 90q --limit 100
python script/benchmark/run_llm_eval_api_call.py --models swiss-ai/Apertus-v1.5-70B --datasets 90q --limit 100

# < ---------------->
# Base depth 1-5; unit scale change
python script/compiler_pipeline_adversarial/make_random_questions_adversarial.py --n 100 --depth-min 1 --depth-max 5 --obstacle unit_scale_change
python script/benchmark/run_llm_eval_api_call.py --models swiss-ai/Apertus-v1.5-70B --datasets 90q --limit 100
python script/benchmark/run_llm_eval_api_call.py --models swiss-ai/Apertus-v1.5-70B --datasets 90q --limit 100

# < ---------------->
# Base depth 1-5; useless info
python script/compiler_pipeline_adversarial/make_random_questions_adversarial.py --n 100 --depth-min 1 --depth-max 5 --obstacle useless_info,
python script/benchmark/run_llm_eval_api_call.py --models swiss-ai/Apertus-v1.5-70B --datasets 90q --limit 100
python script/benchmark/run_llm_eval_api_call.py --models swiss-ai/Apertus-v1.5-70B --datasets 90q --limit 100