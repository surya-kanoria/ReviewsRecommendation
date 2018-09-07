import pandas as pd
from topic_model import *
from utils.text_utils import TextUtils

# Format of file:reviewId | productId | userId | title | text | rating | createStamp | upvote | downvote
reviews_file_path = '/Users/surya.kanoria/Flipkart/Repositories/ugc-labs/Personalised Review Recommendation/data/mob_reviews_100000.csv'
reviews_data = pd.read_csv(reviews_file_path)
train_set = reviews_data.truncate(0, 1000)
max_vocab_size = 50000
num_of_topics = 8
sampler = SentimentBasedLDA(alpha=0.1, beta=0.01, gamma=1, num_of_sentiments=2,
                            text_utils=TextUtils(max_vocab_size=max_vocab_size, vectorizer=None),
                            num_of_topics=num_of_topics)
train_reviews = list(train_set.text)
train_items = list(train_set.productId.unique())
train_users = list(train_set.userId.unique())
print ("Reviews:", len(train_reviews))
print ("Items:", len(train_items))
print ("Users:", len(train_users))

sampler.run(train_reviews, 5)
print ("Topic Distribution", sampler.get_topics_distribution_per_document())

test_set = reviews_data.truncate(1000, 1500)
test_reviews = list(test_set.text)
test_items = list(test_set.productId.unique())
test_users = list(test_set.userId.unique())
sampler.test_model(test_reviews, 2)
