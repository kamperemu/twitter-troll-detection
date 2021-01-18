# Twitter Troll Detection

this project is not a commercial product currently, its mostly used for research work.
## Abstract
In the present age of social media websites, trolls are inevitable. Trolls are people or bots who spread misinformation on social media to push their political agenda. These trolls use fake accounts to stay anonymous and influence public opinion. There is a problem with manual detection of trolls - humans too find it difficult to detect trolls and it isn't possible for humans to  check millions of messages that are uploaded everyday. Therefore, using AI to detect such trolls is much easier and efficient. Different models like SVM, neural networks, logistic regression can be used. These can also be used with pre-trained Natural Language Processing(NLP) models and combine it with the former models. These models are used to find out whether or not the message content is a troll or not by discovering patterns in these tweets We can then choose the most accurate model for direct implementation.
## Instructions

You need to install the following modules on python 3.8.5 - tensorflow,numpy,pickle,matplotlib,seaborn,sklearn,pandas,nltk

extract the data.rar file in order to use the main dataset

You can change the smallerdata.json file to data.json file if you want to try out the faster dataset (The smallerdata.json has only 30,000 tweets while the main file has 1,000,000 tweets). The smallerdata dataset will take significantly less time than the main dataset.

NOTE: You can create your own dataset by creating a csv (with encoding unicode UTF-8) with headers content and label where the content has tweets and the label says whether the data is a troll or not (the label doesn't necessarily have to classify trolls it could classify whatever you want)

The saved models are of the smaller dataset so for the better implementation you need to to run the models at least once

## file name abbreviations
nn - neural network

nb - naive bayes

svm - support vector machine

lr - logistic regression

## Recommendations

https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi

https://www.youtube.com/watch?v=yIYKR4sgzI8

https://www.youtube.com/watch?v=O2L2Uv9pdDA

https://www.youtube.com/watch?v=efR1C6CvhmE

https://www.youtube.com/watch?v=Kdsp6soqA7o

## References
### 1. Modules

https://www.nltk.org/

https://scikit-learn.org/

https://matplotlib.org/

https://seaborn.pydata.org/

https://numpy.org/

### 2. Datasets

https://www.kaggle.com/crowdflower/twitter-user-gender-classification

https://www.kaggle.com/vikasg/russian-troll-tweets?ref=hackernoon.com 

https://www.kaggle.com/kazanova/sentiment140

https://www.kaggle.com/dataturks/dataset-for-detection-of-cybertrolls

https://www.data.gouv.fr/fr/datasets/credibility-corpus-with-several-datasets-twitter-web-database-in-french-and-english/?ref=hackernoon.com#_

https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JBXKFD&version=2.2

https://github.com/fivethirtyeight/russian-troll-tweets

### 3. Algorithms

https://www.kaggle.com/lakshmi25npathi/sentiment-analysis-of-imdb-movie-reviews

https://goo.gle/tfw-sarcembed

### 4. research

https://www.dsayce.com/social-media/tweets-day/

https://www.investopedia.com/terms/m/machine-learning.asp

https://monkeylearn.com/blog/what-is-tf-idf/

https://www.analyticsvidhya.com/blog/2020/02/quick-introduction-bag-of-words-bow-tf-idf/

https://www.machinelearningplus.com/predictive-modeling/how-naive-bayes-algorithm-works-with-example-and-full-code/

https://machinelearningmastery.com/logistic-regression-for-machine-learning/#:~:text=Logistic%20regression%20uses%20an%20equation,an%20output%20value%20(y).

https://towardsdatascience.com/support-vector-machine-introduction-to-machine-learning-algorithms-934a444fca47
