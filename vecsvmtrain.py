import pandas as pd
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
import pickle
from gensim.models import Word2Vec


# loading and modifying the tweets for the algorithm
tweets = pd.read_csv("datasets/test.csv")

tweets['content'] = tweets['content'].astype('str')
tweets['content'].dropna(inplace=True)
tweets['content'] = [entry.lower() for entry in tweets['content']]



# tokenization algorithm
tweets['content']= [word_tokenize(entry) for entry in tweets['content']]

tagMap = defaultdict(lambda : wn.NOUN)
tagMap['J'] = wn.ADJ
tagMap['V'] = wn.VERB
tagMap['R'] = wn.ADV

# processing the data for a better tokenization
for index,entry in enumerate(tweets['content']):
    finalWords = []
    wordLemmatizer = WordNetLemmatizer()
    for word, tag in pos_tag(entry):
        if word not in stopwords.words('english') and word.isalpha():
            wordFinal = wordLemmatizer.lemmatize(word,tagMap[tag[0]])
            finalWords.append(wordFinal)
    tweets.loc[index,'textFinal'] = str(finalWords)

xtrain, xtest, ytrain, ytest = model_selection.train_test_split(tweets['textFinal'],tweets['label'],test_size=0.3)

all_words = [word_tokenize(entry) for entry in tweets['textFinal']]

word2vec = Word2Vec(all_words, min_count=2)

# training the classifier algorithm
clf = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
clf.fit(xtrain,ytrain)
pred = clf.predict(xtest)
print("SVM Accuracy Score -> ",accuracy_score(pred, ytest)*100)
