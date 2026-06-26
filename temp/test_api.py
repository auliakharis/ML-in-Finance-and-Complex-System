import openai
import os

client = openai.Client(
    api_key=os.environ.get("CSCS_SERVING_API"),
    base_url="https://api.swissai.svc.cscs.ch/v1"
)
res = client.chat.completions.create(
    model="swiss-ai/Apertus-8B-Instruct-2509",
    messages=[{"content": "Do you know Aulia Kharis Rakhmasari?", "role": "user"}],
    stream=True,
)

for chunk in res:
    if len(chunk.choices) > 0 and chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
