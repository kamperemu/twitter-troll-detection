# Twitter Troll Detection

## [Research report](https://drive.google.com/file/d/1U5aCkfYpXaT2ELI5aioI8qicHOSjgPHc/view?usp=share_link)

this project is not a commercial product currently, its mostly used for research work. 

The project is also a work in progress so there may be differences between versions. (so there may be discrepancies in the research paper and the algorithms)

## Abstract

In the present age of social media websites, trolls are inevitable. Trolls are people or bots who spread misinformation on social media to push their political agenda. These trolls use fake accounts to stay anonymous and influence public opinion. There is a problem with manual detection of trolls - humans too find it difficult to detect trolls and it isn't possible for humans to  check millions of messages that are uploaded everyday. Therefore, using AI to detect such trolls is much easier and efficient. Different models like SVM, neural networks, logistic regression can be used. These can also be used with pre-trained Natural Language Processing(NLP) models and combine it with the former models. These models are used to find out whether or not the message content is a troll or not by discovering patterns in these tweets We can then choose the most accurate model for direct implementation.

## Instructions

I used python 3.9.13 for this so install that if you can for tested compatibility. Also make sure its 64 bit version of python or else some packages will not install.

Use "pip install -r requirements.txt" in a cli to install all packages required. (Optional: You can also create a venv before installing the packages)

## file name abbreviations
nn - neural network

nb - naive bayes

svm - support vector machine

lr - logistic regression

## References

### 1. Modules

https://www.nltk.org/

https://scikit-learn.org/

https://matplotlib.org/

https://seaborn.pydata.org/

https://numpy.org/

https://pandas.pydata.org/

### 2. Datasets

https://www.kaggle.com/vikasg/russian-troll-tweets?ref=hackernoon.com 

https://www.kaggle.com/kazanova/sentiment140

### 3. Algorithms

https://www.kaggle.com/lakshmi25npathi/sentiment-analysis-of-imdb-movie-reviews

https://goo.gle/tfw-sarcembed
