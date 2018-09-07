import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer

porterStemmer = PorterStemmer()


class TextUtils:
    def __init__(self, vectorizer, max_vocab_size):
        if vectorizer:
            self.vectorizer = vectorizer
        else:
            self.vectorizer = CountVectorizer(analyzer="word",
                                              tokenizer=None,
                                              preprocessor=None,
                                              stop_words="english",
                                              max_features=max_vocab_size)

    def clean_up_text(self, document):
        letters_only = re.sub("[^a-zA-Z]", " ", document)
        words = letters_only.lower().split()
        stops = set(stopwords.words("english"))
        meaningful_words = [porterStemmer.stem(w) for w in words if w not in stops]
        return " ".join(meaningful_words)

    def get_word_matrix(self, documents):
        processed_reviews = []
        i = 0
        for document in documents:
            if (i + 1) % 1000 == 0:
                print "Review %d of %d" % (i + 1, len(documents))
            processed_reviews.append(self.clean_up_text(document))
            i += 1

        train_data_features = self.vectorizer.fit_transform(processed_reviews)
        word_occurence_matrix = train_data_features.toarray()
        return word_occurence_matrix

    def get_vectorizer(self):
        return self.vectorizer

    def transform(self, documents):
        processed_reviews = []
        i = 0
        for document in documents:
            if (i + 1) % 1000 == 0:
                print "Review %d of %d" % (i + 1, len(documents))
            processed_reviews.append(self.clean_up_text(document))
            i += 1
        test_data_features = self.vectorizer.transform(processed_reviews)
        return test_data_features.toarray()
