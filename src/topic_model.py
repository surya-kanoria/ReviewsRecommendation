import numpy as np

from nltk.corpus import sentiwordnet as swn


class SentimentBasedLDA:

    def __init__(self, num_of_topics, alpha, beta, gamma, text_utils, num_of_sentiments=2):
        """
        Initializes the model parameters
        :param text_utils: src.utils.text : TextUtils : should implement function get_word_matrix which represents word vector for documents
        :param num_of_topics: Number of topics in the model
        :param alpha: Hyperparameter for Dirichlet prior on topic distribution
        per document
        :param beta:  Hyperparameter for Dirichlet prior on vocabulary distribution
        per (topic, sentiment) pair
        :param gamma: Hyperparameter for Dirichlet prior on sentiment distribution
        per (document, topic) pair
        :param num_of_sentiments: Number of sentiments per document (default = 2)
        """
        self.num_of_topics = num_of_topics
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.num_of_sentiments = num_of_sentiments
        self.text_utils = text_utils

    @staticmethod
    def sample_from_dirichlet(alpha):
        """
           Sample from a Dirichlet distribution
           alpha: Dirichlet distribution parameter (of length d)
           Returns:
           x: Vector (of length d) sampled from dirichlet distribution

           """
        return np.random.dirichlet(alpha)

    @staticmethod
    def sample_from_categorical(theta):
        """
        Samples from a categorical/multinoulli distribution
        theta: parameter (of length d)
        Returns:
        x: index ind (0 <= ind < d) based on probabilities in theta
        """
        theta = theta / np.sum(theta)
        return np.random.multinomial(1, theta).argmax()

    @staticmethod
    def word_indices(word_occurence_vector):
        """
        Turn a document vector of size vocab_size to a sequence
        of word indices. The word indices are between 0 and
        vocab_size-1. The sequence length is equal to the document length.
        """
        for idx in word_occurence_vector.nonzero()[0]:
            for i in range(int(word_occurence_vector[idx])):
                yield idx

    def _initialize_test(self, documents):
        self.word_matrix = self.text_utils.get_word_matrix(documents)
        number_of_docs, vocabulary_size = self.word_matrix.shape

        # Pseudocounts
        self.n_dt = np.zeros((number_of_docs, self.num_of_topics))
        self.n_dts = np.zeros((number_of_docs, self.num_of_topics, self.num_of_sentiments))
        self.n_d = np.zeros(number_of_docs)
        self.n_vts = np.zeros((vocabulary_size, self.num_of_topics, self.num_of_sentiments))
        self.n_ts = np.zeros((self.num_of_topics, self.num_of_sentiments))
        self.topics = {}
        self.sentiments = {}
        self.priorSentiment = {}

        alpha_vector = self.alpha * np.ones(self.num_of_topics)
        gamma_vector = self.gamma * np.ones(self.num_of_sentiments)

        for i, word in enumerate(self.text_utils.get_vectorizer().get_feature_names()):
            synsets = swn.senti_synsets(word)
            posScore = np.mean([s.pos_score() for s in synsets])
            negScore = np.mean([s.neg_score() for s in synsets])
            if posScore >= 0.1 and posScore > negScore:
                self.priorSentiment[i] = 1
            elif negScore >= 0.1 and negScore > posScore:
                self.priorSentiment[i] = 0

        print (self.priorSentiment)

        for d in range(number_of_docs):
            topic_distribution = self.sample_from_dirichlet(alpha_vector)
            sentiment_distribution = np.zeros(
                (self.num_of_topics, self.num_of_sentiments))
            for t in range(self.num_of_topics):
                sentiment_distribution[t, :] = self.sample_from_dirichlet(gamma_vector)
            for i, w in enumerate(self.word_indices(self.word_matrix[d, :])):
                t = self.sample_from_categorical(topic_distribution)
                s = self.sample_from_categorical(sentiment_distribution[t, :])

                self.topics[(d, i)] = t
                self.sentiments[(d, i)] = s
                self.n_dt[d, t] += 1
                self.n_dts[d, t, s] += 1
                self.n_d[d] += 1
                self.n_vts[w, t, s] += 1
                self.n_ts[t, s] += 1

    def conditional_distribution(self, d, v):
        """
        Calculates the (topic, sentiment) probability for word v in document d
        Returns:    a matrix (numTopics x numSentiments) storing the probabilities
        """
        probabilities_ts = np.ones((self.num_of_topics, self.num_of_sentiments))
        first_factor = (self.n_dt[d] + self.alpha) / \
                       (self.n_d[d] + self.num_of_topics * self.alpha)
        second_factor = (self.n_dts[d, :, :] + self.gamma) / \
                        (self.n_dt[d, :] + self.num_of_sentiments * self.gamma)[:, np.newaxis]
        third_factor = (self.n_vts[v, :, :] + self.beta) / \
                       (self.n_ts + self.n_vts.shape[0] * self.beta)
        probabilities_ts *= first_factor[:, np.newaxis]
        probabilities_ts *= second_factor * third_factor
        probabilities_ts /= np.sum(probabilities_ts)
        return probabilities_ts

    # http://u.cs.biu.ac.il/~89-680/darling - lda.pdf
    # Algorithm - 1: Gibbs sampling
    def run(self, documents, max_iterations=30):
        """
        Runs Gibbs sampler for sentiment-LDA
        """
        self._initialize_test(documents)
        num_docs, vocabulary_size = self.word_matrix.shape
        for iteration in range(max_iterations):
            print "Starting iteration %d of %d" % (iteration + 1, max_iterations)
            for d in range(num_docs):
                for i, word in enumerate(self.word_indices(self.word_matrix[d, :])):
                    t = self.topics[(d, i)]
                    s = self.sentiments[(d, i)]
                    self.n_dt[d, t] -= 1
                    self.n_d[d] -= 1
                    self.n_dts[d, t, s] -= 1
                    self.n_vts[word, t, s] -= 1
                    self.n_ts[t, s] -= 1

                    probabilities_ts = self.conditional_distribution(d, word)
                    if word in self.priorSentiment:
                        s = self.priorSentiment[word]
                        t = self.sample_from_categorical(probabilities_ts[:, s])
                    else:
                        ind = self.sample_from_categorical(probabilities_ts.flatten())
                        t, s = np.unravel_index(ind, probabilities_ts.shape)

                    self.topics[(d, i)] = t
                    self.sentiments[(d, i)] = s
                    self.n_dt[d, t] += 1
                    self.n_d[d] += 1
                    self.n_dts[d, t, s] += 1
                    self.n_vts[word, t, s] += 1
                    self.n_ts[t, s] += 1

        print "training done"

    def get_topics_distribution_per_document(self):
        return self.n_dts

    def test_model(self, reviews, max_iterations=5):
        new_word_matrix = self.text_utils.transform(reviews)
        np.append(self.word_matrix, new_word_matrix)
        num_docs, vocabulary_size = self.word_matrix.shape
        for iteration in range(max_iterations):
            print "Starting iteration %d of %d" % (iteration + 1, max_iterations)
            for d in range(num_docs):
                for i, word in enumerate(self.word_indices(self.word_matrix[d, :])):
                    t = self.topics[(d, i)]
                    s = self.sentiments[(d, i)]
                    self.n_dt[d, t] -= 1
                    self.n_d[d] -= 1
                    self.n_dts[d, t, s] -= 1
                    self.n_vts[word, t, s] -= 1
                    self.n_ts[t, s] -= 1

                    probabilities_ts = self.conditional_distribution(d, word)
                    if word in self.priorSentiment:
                        s = self.priorSentiment[word]
                        t = self.sample_from_categorical(probabilities_ts[:, s])
                    else:
                        ind = self.sample_from_categorical(probabilities_ts.flatten())
                        t, s = np.unravel_index(ind, probabilities_ts.shape)

                    self.topics[(d, i)] = t
                    self.sentiments[(d, i)] = s
                    self.n_dt[d, t] += 1
                    self.n_d[d] += 1
                    self.n_dts[d, t, s] += 1
                    self.n_vts[word, t, s] += 1
                    self.n_ts[t, s] += 1
        print "running on test set over"