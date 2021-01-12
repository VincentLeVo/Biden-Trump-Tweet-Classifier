# Biden-Trump-Tweet-Classifier
This classifier predicts whether the tweet was composed from either Donald Trump or Joe Biden

## What it does

 Given a tweet, this classifier predicts whether the tweet was composed from either Donald Trump or Joe Biden. The output is the training error and testing error. 
## Classifiers
I implemented the following classifiers:

1. The Random Forest classifier - using the mini index as a splitting score and where the K-means algorithm used to quantize each feature, determining the threshold search over for each feature.
2. Naive Bayes 
3. K-Nearest neighbourhoods with cosine similarity as the distance metric
4. Stacking Ensembling 
## Technologies Used
I was given a dataset consisting of tweets from the two USA Presidential candidates of 2020 - Donald Trump and Joe Biden. The features of each word in each word was extracted using the word2vec algorithm then combined by averaging to form a fixed-length feature vector per tweet. 
From that point on, I used various machine learning models to implement and classify the tweet. 

## How to Use:

1. Install Python 3.- -
2. cd into the 'code' directory
3.  
Depending on the classifier :
```
python main -q 1
python main -q 2
python main -q 3
python main -q 4
```
