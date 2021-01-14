import pandas as pd
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score
import pickle


# loading and modifying the tweets for the algorithm
tweets = pd.read_csv("datasets/test.csv")
tweets = tweets.sample(frac = 1)

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

# encoding the data
Encoder = LabelEncoder()
ytrain = Encoder.fit_transform(ytrain)
ytest = Encoder.fit_transform(ytest)
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(tweets['textFinal'])
xtrain = Tfidf_vect.transform(xtrain)
xtest = Tfidf_vect.transform(xtest)

# training the classifier algorithm
clf = naive_bayes.MultinomialNB()
clf.fit(xtrain,ytrain)
pickle.dump(clf, open("savedModel/nb/basic/model.sav","wb"))
pickle.dump(Tfidf_vect, open("savedModel/nb/basic/Tfidf.sav","wb"))

pred = clf.predict(xtest)
print("Naive Bayes Accuracy Score -> ",accuracy_score(pred, ytest)*100)

