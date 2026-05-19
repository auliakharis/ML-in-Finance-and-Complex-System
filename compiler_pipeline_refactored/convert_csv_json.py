import csv
import json

data = []  # ← missing this

with open("output/random_questions_90.csv", 'r', encoding='utf-8') as csv_file:
    reader = csv.DictReader(csv_file)
    for row in reader:
        data.append(dict(row))

with open("output/random_questions_90.json", 'w', encoding='utf-8') as json_file:  # ← fix extension
    json.dump(data, json_file, indent=4)

print(f"Done! {len(data)} rows converted.")