import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score


base_path = "BBC/"

# getting file distribution
classes = {}
max_count = 0
total_count = 0
for item in os.listdir(base_path):
    path = os.path.join(base_path, item)
    if(os.path.isdir(path)):
        count = len([file for file in os.listdir(path) if os.path.isfile(os.path.join(path, file))])
        if count > max_count:
            max_count = count
        classes.update({item : count})
        total_count += count



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

# ... renaming to avoid confusion
y_true = y_test


# Train and predict MultinomialNB
def train_and_predict_MNB(clf, X_test, X_train, y_train):
    # Training multinomial Naive Bayes
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred



# Outputs information about the performance of our model to a file
def analyze_performance(name, y_true, y_pred, corpus, clf, classes, append=False):
    write_mode = "a" if append else "w"
    with open("bbc-performance.txt", write_mode) as f:
        nb_confusion = confusion_matrix(y_true, y_pred)
        nb_f1_measure = classification_report(y_true, y_pred, target_names=corpus.target_names, digits=3)
        nb_f1_score_macro = f1_score(y_true, y_pred, average="macro")
        nb_f1_score_weighted = f1_score(y_true, y_pred, average="weighted")
        nb_acc = accuracy_score(y_true, y_pred)
        nb_word_token_total = clf.feature_count_.sum()
        f.write("================================================================\n")
        f.write(f"(a) {name}\n")
        f.write("\n(b) Confusion matrix\n")
        f.write(f"{nb_confusion}\n")
        f.write(f"\n(c) {nb_f1_measure}\n")
        f.write(f"\n(d) \nAccuracy of model: {nb_acc:.2%}\n")
        f.write(f"Macro-average F1 of model: {nb_f1_score_macro:.2%}\n")
        f.write(f"Weighted-average F1 of model: {nb_f1_score_weighted:.2%}\n")
        f.write("\n(e) Priors\n")
        for i in classes:
            f.write(f"{i}: {classes[i]/total_count:.2%}\n")

        f.write(f"\n(f) Size of vocabulary: {clf.n_features_in_}\n")
        f.write("\n(g) Number of world-tokens per class\n")

        for i, name in enumerate(classes):
            f.write(f"{i}. {name}: {clf.feature_count_[i].sum():.0f}\n")

        f.write(f"\n(h) Number of world-tokens in entire corpus: {nb_word_token_total:.0f}\n")

        f.write("\n(i) Number and percentage of words with a frequency of zero per class\n")
        for i, name in enumerate(classes):
            c = np.count_nonzero(clf.feature_count_[i]==0)
            f.write(f"{i}. {name}: {c} ({c / clf.feature_count_[i].sum():.2%})\n")

        f.write("\n(j) Number and percentage of words with a frequency of one in entire corpus:\n")
        c = np.count_nonzero(clf.feature_count_==1)
        f.write(f"{c} ({c / nb_word_token_total:.2%})\n")

        f.write("\n(k) Two favorite words in vocabulary, and their log-prob\n")
        # not sure what best way to do this is
        arr = vectorizer.get_feature_names_out()
        loc1 = np.where(arr == "zombie")[0][0]
        log_prob1 = clf.feature_log_prob_[:,loc1]
        loc2 = np.where(arr == "potato")[0][0]
        log_prob2 = clf.feature_log_prob_[:,loc2]

        f.write("\"Zombie\":\n")
        for i, name in enumerate(classes):
            f.write(f"{i}. {name}: {log_prob1[i]}\n")

        f.write("\n\"Potato\":\n")
        for i, name in enumerate(classes):
            f.write(f"{i}. {name}: {log_prob2[i]}\n")

        f.write("================================================================\n")

# Performing analysis
def main():
    clf = MultinomialNB()
    y_pred = train_and_predict_MNB(clf, X_test, X_train, y_train)
    analyze_performance("MultinomialNB default values, try 1", y_true, y_pred, corpus, clf, classes)

    clf = MultinomialNB()
    y_pred = train_and_predict_MNB(clf, X_test, X_train, y_train)
    analyze_performance("MultinomialNB default values, try 2", y_true, y_pred, corpus, clf, classes, True)

    clf = MultinomialNB(alpha=0.0001)
    y_pred = train_and_predict_MNB(clf, X_test, X_train, y_train)
    analyze_performance("MultinomialNB default values, try 3", y_true, y_pred, corpus, clf, classes, True)

    clf = MultinomialNB(alpha=0.9)
    y_pred = train_and_predict_MNB(clf, X_test, X_train, y_train)
    analyze_performance("MultinomialNB default values, try 4", y_true, y_pred, corpus, clf, classes, True)

if __name__ == "__main__":
    main()
