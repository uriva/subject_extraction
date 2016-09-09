# CoCycles proprietary and confidential (c) 2015
# 
# NLP Candidate Evaluation 
# =======================
#
# The following is a short rudimentary machine learning task that you'll be
# asked to perform in Python.  Take all the time you need until we kick you out
# :) Before you begin, make sure that you have a machine with internet
# connection and Python.
#
# Only edit this file. Answer each question in a comment below the question.  To
# be clear - you're supposed to write all the code yourself. Do not copy and
# paste code.
#
# The dataset consists of a subset of the classical 20_newsgroups dataset. There
# are a few thousand newsgroup discussions, classified into four topics. Your
# goal is to write a semi-supervised system to classify newsgroups into topics.

#################################################################################

from sklearn.datasets import fetch_20newsgroups


categories = [
    'alt.atheism',
    'talk.religion.misc',
    'comp.graphics',
    'sci.space',
]
                                                                                                                                                       
dataset = fetch_20newsgroups(subset='all', categories=categories,
                             shuffle=True, random_state=42)
                             
dataset.data #this is the data
dataset.target #these are the class labels

#################################################################################

#
# Tasks 
# =====
#
# 1. Load the training dataset. In a remark below this line describe both
# datasets in a few words.

"""
dataset.data has a list of newsgroup messages, each with a header with fields such as "from".
dataset.target has labels corresponding to the 4 subjects of discussion, encoded as numbers 0-3 inclusive.
"""

# 2. Find the 1000 most common words in the dataset. (Bonus: try to ignore case,
# plurality, etc.  There are additional pre-processing steps that you should
# take in order to improve results. Try to think of a few.  write what you chose
# to ignore)

def get_top_common_words(data=dataset.data, n=1000):
    return [x[0] for x in word_histogram(data)[:n]]

# The list contains some number, but mostly words (didn't see a reason to treat them differently).
#
# 3. Form term document matrix, whose (i,j)-th entry is the number of
# occurrences of word i in document j.

def get_term_matrix():
    terms = get_top_common_words()
    return make_termdoc_matrix(terms)

# 4. Set aside 500 random documents as a held-out testset. Based on the training
# set, build a classifier to classify an un-labeled document into one of the
# four topics.  You can use any features you like. For example, you can use a
# term document matrix as above, but you may also choose to use different
# features.  Explain all algorithm and design choices you made along the way:
# why this classifier, why these features, etc.  Train your classifier on the
# training set.
# Report the results of your classifier on the test set.

"""
I chose decision trees with the term-doc matrix.
Decision trees are fast: one can run them on all the complete word matrix and finish in a few minutes.
They are relatively simple to set up, no parameters for the basic usage.
For this task, I assumed that the occurrence of certain terms is highly indicative of the subject, and it suffices to find these.
In other words, no complex functions joining several terms should be learned.
Using a document to n-grams matrix (say up to 3-grams) would probably improve on this, as some phrases are multi-worded.
In addition it is possible to look at mailing addresses, and give subject lines some extra weight.
I didn't implement these suggestions because of lack of time.
Anyway the performance is reasonable but meh: 78% correct answers.
"""

def classifier(data=dataset.data, target=dataset.target):
    # Randomize data (first 500 will be treated as test).
    data_and_target = zip(data, target)
    import random
    random.shuffle(data_and_target)
    data = [x[0] for x in data_and_target]
    target = [x[1] for x in data_and_target]

    # Extract the doc-term matrix terms (using only the training data).
    feature_tokens = get_top_common_words(data=data[500:], n=1000000)

    # Make the matrix over all data and run the classifier.
    all_vectors = make_termdoc_matrix(feature_tokens=feature_tokens, data=data)
    train_and_eval_classifier(all_vectors[500:], target[500:], all_vectors[:500], target[:500])


################################################################################
# helper methods:
################################################################################

def tokenize(text):
    import re
    # Change all known word separators to spaces.
    word_sep = r'[/\-.@,|:\n\t\\]'
    text = re.sub(word_sep, ' ', text)
    # Remove everything except alphanumeric and spaces.
    text = re.sub(r'[^A-Za-z0-9\s]+', '', text)
    tokens = text.split(' ')
    # This will remove double spaces.
    tokens = filter(len, tokens)
    tokens = [token.lower() for token in tokens]
    return tokens
    
def word_histogram(data):
    unique_words = {}

    # Tokenize and make histogram.
    for text in data:
        tokens = tokenize(text)
        for token in tokens:
            if token not in unique_words:
                unique_words[token] = 0
            unique_words[token] += 1
    
    # Stem words to ignore commong morphologies.
    def stem(word):
        stem_options = []
        for prefix in ['re']:
            if word.startswith(prefix):
                stem_options.append(word[len(prefix):])
        for suffix in ['s', 'ing', 'd', 'ed', 'ly', 'er']:
            if word.endswith(prefix):
                stem_options.append(word[:len(suffix)])
        return stem_options
    
    # Reduce histogram using stem.
    for word, frequency in unique_words.items():
        for stemmed in stem(word):
            if (stemmed in unique_words):
                unique_words[stemmed] += frequency
                del unique_words[word]
                break
    
    return sorted(unique_words.items(), key=lambda x: x[1], reverse=True)



# I used these functions to get the set of words for the `stem` function.
def print_morphology():
    word_hist = word_histogram(dataset.data)
    unique_words = [x[0] for x in word_hist]
    prefixes = Morphology.find_common_prefixes(unique_words)
    suffixes = Morphology.find_common_suffixes(unique_words)
    print(prefixes[:100], suffixes[:100])

class Morphology:
    # Finds commong suffixes such as 'ing' and 'ion'.
    # We expect that going over pairs where at least one word is common will suffice.
    @staticmethod
    def find_common_suffixes(unique_words, top_n=1000):
        suffixes = {}
        for i in range(min(len(unique_words), top_n)):
            word1 = unique_words[i]
            # There is no importance to order of elements in each pair, so we can start with i.
            for j in range(i + 1, len(unique_words)):
                word2 = unique_words[j]
                suffix = Morphology.words_diff(word1, word2)
                # Digits are not intersting.
                if not suffix or suffix.isdigit():
                    continue
                if suffix not in suffixes:
                    suffixes[suffix] = 0
                suffixes[suffix] += 1
        return sorted(suffixes.items(), key=lambda x: x[1], reverse=True)

    # Kinda stupid but works.
    @staticmethod
    def find_common_prefixes(unique_words, top_n=1000):
        reversed_words = [x[::-1] for x in unique_words]
        prefixes_reversed = Morphology.find_common_suffixes(reversed_words, top_n)
        return [(x[0][::-1], x[1]) for x in prefixes_reversed]

    @staticmethod
    def words_diff(w1, w2):
        i = 0
        while i < len(w1) and i < len(w2) and w1[i] == w2[i]:
            i += 1
        if i > 0:
            if i == len(w1):
                return w2[i:]
            if i == len(w2):
                return w2[i:]

# feature_tokens maps token to index.
def make_termdoc_matrix(feature_tokens, data):
    token_to_index = {}
    for token in feature_tokens:
        token_to_index[token] = len(token_to_index)
    all_vectors = []
    for text in data:
        vector = [0]*len(feature_tokens)
        for token in tokenize(text):
            if token in feature_tokens:
                vector[token_to_index[token]] += 1
        all_vectors.append(vector)
    return all_vectors

def train_and_eval_classifier(train_data, train_target, test_data, test_target):
    from sklearn import tree
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train_data, train_target)
    predicted_test_target = clf.predict(test_data)
    right = 0
    wrong = 0
    for i in range(len(test_data)):
        if predicted_test_target[i] == test_target[i]:
            right += 1
        else:
            wrong += 1
    print(right, wrong)
