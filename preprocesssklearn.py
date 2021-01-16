import re
import nltk
from nltk.tokenize.toktok import ToktokTokenizer



def removespchar(text):
    pattern=r'[^a-zA-Z0-9\s]'
    text=re.sub(pattern,'',text)
    return text

def stemmer(text):
    ps=nltk.porter.PorterStemmer()
    text= ' '.join([ps.stem(word) for word in text.split()])
    return text

def removestopwords(text, islowercase=True):
    tokenizer=ToktokTokenizer()
    stopwordList=nltk.corpus.stopwords.words('english')
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if islowercase:
        filteredTokens = [token for token in tokens if token not in stopwordList]
    else:
        filteredTokens = [token for token in tokens if token.lower() not in stopwordList]
    filteredText = ' '.join(filteredTokens)    
    return filteredText
