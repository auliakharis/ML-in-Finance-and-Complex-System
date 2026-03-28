import json
import pandas as pd

input_file = "output/final_90_questions_depth10.json"
output_file = "output/final_90_questions_depth10.csv"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

df = pd.json_normalize(data)
df.to_csv(output_file, index=False)

print(f"Saved CSV to {output_file}")