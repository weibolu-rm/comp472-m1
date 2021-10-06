import os
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
# nb_correct = (y_test == clf.predict(X_test)).sum()
# nb_incorrect = y_test.size - nb_correct
# print(f"{nb_correct} documents correctly classified\n{nb_incorrect} documents incorrectly classified")

# renaming to avoid confusion
y_true = y_test
y_pred = clf.predict(X_test)
print(f"Accuracy score: {accuracy_score(y_true, y_pred)}")



# writing to file
with open("bbc-performance.txt", "w") as f:
    nb_confusion = confusion_matrix(y_true, y_pred)
    nb_f1_measure = classification_report(y_true, y_pred)
    nb_f1_score_macro = f1_score(y_true, y_pred, average="macro")
    nb_f1_score_weighted = f1_score(y_true, y_pred, average="weighted")
    nb_acc = accuracy_score(y_true, y_pred)
    f.write("================================================================\n")
    f.write("(a) MultinomialNB default values, try 1\n")
    f.write("(b) Confusion matrix\n")
    f.write(f"{nb_confusion}\n")
    f.write(f"(c) {nb_f1_measure}\n")
    f.write(f"(d) \nAccuracy of model: {nb_acc}\n")
    f.write(f"Macro-average F1 of model: {nb_f1_score_macro}\n")
    f.write(f"Weighted-average F1 of model: {nb_f1_score_weighted}\n")



    f.write("================================================================\n")

