import csv
import json

with open("output/random_questions_90_alicia.csv", mode="r", newline="", encoding="utf-8") as csvfile:
    data = list(csv.DictReader(csvfile))
    
with open("output/json_format/questions_multfact1000000_depth20.json", mode="w", encoding="utf-8") as jsonfile:
    json.dump(data, jsonfile, indent=4)
