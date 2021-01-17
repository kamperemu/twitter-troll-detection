import csv
import json
data = []
with open("test.csv", encoding='utf-8') as csvf:
	csvReader = csv.DictReader(csvf)
	for rows in csvReader:
		data.append(rows)
	
with open("test.json", 'w', encoding='utf-8') as jsonf:
	jsonf.write(json.dumps(data,indent=4))
