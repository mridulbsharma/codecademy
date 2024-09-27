# Project 3: Web APIs & NLP

# Problem Statement

The goal is to build a binary classification model that can accurately distinguish between cryptocurrency-based posts and stock-based posts from the famous wallstreetbets and CryptoMoonShots subreddits.

# Questions

1. Given a choice between r/wallstreetbets and r/CryptoMoonShots, how accurately can we predict the subreddit of a given post?
2. Which metrics will help us measure successful predictions? 
3. Which model will provide the highest accuracy?
4. Can we ensemble models for an even more improved accuracy?

# Data Dictionary

| Feature         | Type     | Dataset | Description                                      |
|-----------------|----------|---------|--------------------------------------------------|
| **id**          | *object* | All     | The unique ID given to a post by Reddit.         |
| **datetime**    | *object* | All     | Date and time of the post.                       |
| **title**       | *object* | All     | The title of the associated post.                |
| **text**        | *object* | All     | The text content of the post.                    |
| **score**       | *int64*  | All     | Balance of upvotes/downvotes.                    |
| **upvote_ratio**| *float64*| All     | The ratio of upvotes to total votes.             |
| **url**         | *object* | All     | URL of the Reddit post.                          |
| **subreddit**   | *int64*  | All     | Subreddit of origin (1 for r/wallstreetbets, 0 for r/CryptoMoonShots). |
| **has_text**    | *bool*   | All     | Indicates if the post has text content.          |
| **title_len**   | *int64*  | All     | Length of the title.                             |
| **text_len**    | *int64*  | All     | Length of the text content.  


## Data Used
1. cms_controversial_posts.csv
2. cms_merged_posts.csv
3. cms_new_posts.csv
4. cms_top_posts.csv
5. wsb_controversial_posts.csv
6. wsb_merged_posts.csv
7. wsb_new_posts.csv
8. wsb_top_posts.csv
9. combined_df.csv

---

## Requirements
- Python, Jupyter
- Pandas, Numpy, Matplotlib
- Scikit Learn:
   - StandardScaler, Train Test Split, Metrics, Pipeline 
   - CountVectorizer, TfidfVectorizer
   - LogisticRegression, MultinomialNB, RandomForestClassifier
- Nlkt:
   - PorterStemmer, WordNetLemmatizer
   - stopwords, wordnet
   - word_tokenize, sent_tokenize, RegexpTokenizer

---
## Executive Summary
Reddit is an online platform for sharing news, content, and discussions. It features user-created sections called 'subreddits' that cater to a wide range of topics and interests. Members can contribute various types of content, including images, texts, and links, to these subreddits, and other members can express their approval ('upvote') or disapproval ('downvote') of the content.

The model will be trained using Natural Language Processing (NLP) techniques on text data collected from these subreddits, with the help of PRAW. The classification model will help Reddit users identify whether a post is related to cryptocurrency or stocks, which can be valuable for making informed decisions about trading or investment opportunities.

 
 
#### Objectives
1. Gather and prepare data from the wallstreetbets and CryptoMoonShots subreddits using PRAW.
2. Preprocess the text data by cleaning, tokenizing, and vectorizing the posts for NLP analysis.
3. Train and compare two classification models - random forest trees and logistic regression - to predict whether a post is related to cryptocurrency or stocks.
4. Evaluate the performance of the models using appropriate metrics such as accuracy, precision, recall, and F1 score.

#### Methods
    I initialized the PRAW API to access Reddit and selected two subreddits, r/wallstreetbets and r/CryptoMoonShots, from the top subreddits. I then scraped data from these subreddits based on top, new, and controversial posts. After scraping the data, I explored it for missing values and outliers, and combined all the scraped data into larger dataframes based on the subreddit. I binarized the 'subreddit' column in the dataframes for model prediction use, assigning 0 for CryptoMoonShots and 1 for WallStreetBets. Additionally, I created a function to process the title feature by tokenizing, lemmatizing, lowercasing, removing punctuation, and stop words. Subsequently, I combined both dataframes, determined the baseline accuracy, and created models using CountVectorizer (CVEC) and TF-IDF. I also utilized GridSearch to explore which params would improve the model's performance.

#### Findings
  - The Multinomial Naive Bayes model demonstrated a high level of accuracy in predicting the subreddit of origin for a given post.
  - Cross-validation confirmed the model's generalized nature with an accuracy of just under 93%
  - Ensembled model with Naive Bayes and Logisitic Regression gives a slightly more accurate model with accuracy just above 93%.

#### Next steps
  -  Use CNN architecture for classfication and higher accruacy. 

#### Resources
1. wallstreetbets subreddit: https://www.reddit.com/r/wallstreetbets
2. CryptoMoonShots subreddit: https://www.reddit.com/r/CryptoMoonShots
3. PRAW API: https://www.reddit.com/prefs/apps
4. https://github.com/zzeniale/Subreddit-classification