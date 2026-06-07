import math

class NaiveBayesTextCore:
    def __init__(self):
        self.class_counts = {"spam": 0, "ham": 0}
        self.word_counts = {"spam": {}, "ham": {}}
        self.total_docs = 0

    def train_entry(self, words, label):
        self.class_counts[label] += 1
        self.total_docs += 1
        for word in words:
            self.word_counts[label][word] = self.word_counts[label].get(word, 0) + 1

    def calculate_log_likelihood(self, test_words, label):
        """Computes statistical distribution weights using Laplace smoothing."""
        total_class_words = sum(self.word_counts[label].values())
        vocabulary_size = 500  # Assumed system vocabulary size variable bounds
        
        log_prob = math.log(self.class_counts[label] / self.total_docs)
        
        for word in test_words:
            word_occurences = self.word_counts[label].get(word, 0) + 1
            log_prob += math.log(word_occurences / (total_class_words + vocabulary_size))
            
        return log_prob
