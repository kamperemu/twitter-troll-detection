import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle



# loading and modifying the tweets for the algorithm
tweets = pd.read_csv("datasets/test.csv")

tweets['content'] = tweets['content'].astype('str')
tweets['content'].dropna(inplace=True)
tweets['content'] = [entry.lower() for entry in tweets['content']]



# Step - c : Tokenization : In this each entry in the corpus will be broken into set of words
tweets['content']= [word_tokenize(entry) for entry in tweets['content']]

tag_map = defaultdict(lambda : wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV


for index,entry in enumerate(tweets['content']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N) or Verb(V) or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets    
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word,tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    tweets.loc[index,'text_final'] = str(Final_words)


tweets = shuffle(tweets)
xtrain, xtest, ytrain, ytest = model_selection.train_test_split(tweets['text_final'],tweets['label'],test_size=0.3)

Encoder = LabelEncoder()
ytrain = Encoder.fit_transform(ytrain)
ytest = Encoder.fit_transform(ytest)
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(tweets['text_final'])
xtrain_Tfidf = Tfidf_vect.transform(xtrain)
xtest_Tfidf = Tfidf_vect.transform(xtest)

# Classifier - Algorithm - SVM
# fit the training dataset on the classifier
clf = LogisticRegression(max_iter=10000)
clf.fit(xtrain_Tfidf,ytrain)
# predict the labels on validation dataset
pred = clf.predict(xtest_Tfidf)
# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ",accuracy_score(pred, ytest)*100)