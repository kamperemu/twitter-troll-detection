import json
import re
with open("datasets/data.json", 'r') as f:
    tweets = json.load(f)
def removespchar(text):
    pattern=r'[^a-zA-Z0-9\s]'
    text=re.sub(pattern,'',text)
    return text
for i in range(len(tweets)):
    tweets[i]['content']=removespchar(tweets[i]['content'])
with open("data.json", 'w', encoding='utf-8') as jsonf:
	jsonf.write(json.dumps(tweets,indent=4))