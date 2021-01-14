# imports
import pandas as pd
from preprocesssklearn import *
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

tweets = pd.read_csv("datasets/test.csv")

# preprocessing
tweets['content']=tweets['content'].apply(str)
tweets['content']=tweets['content'].apply(denoise_text)
tweets['content']=tweets['content'].apply(remove_special_characters)
tweets['content']=tweets['content'].apply(simple_stemmer)
tweets['content']=tweets['content'].apply(remove_stopwords)


# encoding
tweets = tweets.sample(frac = 1)

xtrain = tweets.content[:int(round(4*(tweets['content'].size)/5))]
xtest = tweets.content[int(round(4*(tweets['content'].size)/5)):]


#Count vectorizer for bag of words
cv=CountVectorizer(min_df=0,max_df=1,binary=False,ngram_range=(1,3))
cvxtrain=cv.fit_transform(xtrain)
cvxtest=cv.transform(xtest)

#Tfidf vectorizer
tv=TfidfVectorizer(min_df=0,max_df=1,use_idf=True,ngram_range=(1,3))
tvxtrain=tv.fit_transform(xtrain)
tvxtest=tv.transform(xtest)

#labeling the sentient data
lb=LabelBinarizer()
labels=lb.fit_transform(tweets['label'])

ytrain = labels[:int(round(4*(labels.size)/5))]
ytest = labels[int(round(4*(labels.size)/5)):]

#training the model
lr=SGDClassifier(loss='hinge',max_iter=500,random_state=42)
lr_bow=lr.fit(cvxtrain,ytrain)
lr_tfidf=lr.fit(tvxtrain,ytrain)

pred = lr_bow.predict(cvxtest)
print("Support Vector Machine Accuracy Score -> ",accuracy_score(pred, ytest)*100)
pred = lr_tfidf.predict(tvxtest)
print("Support Vector Machine Accuracy Score -> ",accuracy_score(pred, ytest)*100)

pickle.dump(lr_bow, open("savedModel/svm/bowmodel.sav","wb"))
pickle.dump(lr_tfidf, open("savedModel/svm/tfidfmodel.sav","wb"))
pickle.dump(tv, open("savedModel/svm/Tfidf.sav","wb"))
pickle.dump(cv, open("savedModel/svm/bow.sav","wb"))
