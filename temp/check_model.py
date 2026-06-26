import openai, os
from dotenv import load_dotenv
load_dotenv()

client = openai.Client(
    api_key=os.environ.get("CSCS_SERVING_API"),
    base_url="https://api.swissai.svc.cscs.ch/v1"
)

models = client.models.list()
for m in models.data:
    print(m.id)

"""
swiss-ai/Apertus-8B-Instruct-2509
cais/HarmBench-Llama-2-13b-cls
moonshotai/Kimi-K2.5
Snowflake/snowflake-arctic-embed-l-v2.0
codellama/CodeLlama-70b-Instruct-hf
deepseek-ai/deepseek-coder-33b-instruct
zai-org/GLM-4.7-Flash
swiss-ai/Apertus-70B-Instruct-2509
openai/gpt-oss-120b-evMj
meta-llama/Llama-3.3-70B-Instruct
mistralai/Voxtral-Small-24B-2507
Qwen/Qwen3-Coder-30B-A3B-Instruct
Qwen/Qwen3.5-27B

"""


