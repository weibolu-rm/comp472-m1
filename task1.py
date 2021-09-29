import os
import matplotlib.pyplot as plt
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split


base_path = "BBC/"

# getting file distribution
classes = {}
max_count = 0
for item in os.listdir(base_path):
    path = os.path.join(base_path, item)
    if(os.path.isdir(path)):
        count = len([file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))])
        if count > max_count:
            max_count = count
        classes.update({item : count})


# Ploting the distribution to BBC-distribution.pdf
plt.plot(list(classes.keys()), list(classes.values()), 'r')
plt.xlabel("Classes")
plt.ylabel("Count")
plt.savefig("BBC-distribution.pdf")


# Loading the corpus
corpus = load_files(base_path, description="BBC dataset", encoding="latin1")

# pre-processing dataset for Naive Bayes
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus.data)


# split dataset 80%/20%
X_train,  X_test , y_train, y_test = train_test_split(X, corpus.target, test_size=0.2, random_state=None)


# Training multinomial Naive Bayes
clf = MultinomialNB()
clf.fit(X_train, y_train)

# Evaluating 
nb_correct = (y_test == clf.predict(X_test)).sum()
nb_incorrect = y_test.size - nb_correct
print(f"{nb_correct} documents correctly classified\n{nb_incorrect} documents incorrectly classified")
