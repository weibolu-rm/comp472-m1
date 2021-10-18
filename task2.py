import pandas as pd
import warnings
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
# a lot of warning from tuning hyper params
warnings.filterwarnings("ignore")

out_path = "out/"
benchmark_file = "drugs-performance.txt"

def load_data():
    drug_path = "data/drug200.csv"
    drug_data = pd.read_csv(drug_path)
    return drug_data

def plot_dist(drug_data):
    drug_classes = {}
    key_list = []

    for item in drug_data['Drug']:
        if drug_classes.get(item) is None:
            drug_classes.update({item : len([x for x in drug_data['Drug'] if x == item])})
            key_list.append(item)

    # sorting
    key_list.sort()
    temp = {}
    for k in key_list:
        temp.update({k : drug_classes.get(k)})

    drug_classes = temp

    # plotting to drug-distribution.pdf
    plt.plot(list(drug_classes.keys()), list(drug_classes.values()), 'r')
    plt.xlabel("Classes")
    plt.ylabel("Count")
    plt.savefig(f"{out_path}drug-distribution.pdf")


def analyze_performance(name, params, y_true, y_pred, append=False):
    write_mode = "a" if append else "w"

    confusion = confusion_matrix(y_true, y_pred)
    f1_measure = classification_report(y_true, y_pred, digits=3, zero_division=0)
    f1_score_macro = f1_score(y_true, y_pred, average="macro")
    f1_score_weighted = f1_score(y_true, y_pred, average="weighted")
    acc = accuracy_score(y_true, y_pred)

    with open(f"{out_path}{benchmark_file}", write_mode) as f:
        f.write("=" * 100)
        f.write(f"\n(a)\nModel: {name}\n")
        f.write(f"Params:\n{params}\n")
        f.write(f"\n(b) Confusion Matrix\n{confusion}\n")
        f.write(f"\n(c)\n{f1_measure}\n")
        f.write(f"\n(d) \nAccuracy of model: {acc:.2%}\n")
        f.write(f"Macro-average F1 of model: {f1_score_macro:.2%}\n")
        f.write(f"Weighted-average F1 of model: {f1_score_weighted:.2%}\n")
        f.write("=" * 100)
        f.write("\n")

# Train and predict
def train_and_predict(clf, X_test, X_train, y_train):
    # Training multinomial Naive Bayes
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred

def main():
    # Loading dataset
    drug_data = load_data()

    # Plot
    plot_dist(drug_data)

    # Nominal to numerical
    df = pd.get_dummies(drug_data, columns=['Sex'])

    # Ordinal to numerical
    df['BP'].replace({'LOW':1, 'NORMAL':2, 'HIGH':3}, inplace=True)
    df['Cholesterol'].replace({'LOW':1, 'NORMAL':2, 'HIGH':3}, inplace=True)
    df =df[['Age', 'BP', 'Cholesterol','Na_to_K', 'Sex_F', 'Sex_M', 'Drug']] # re-ordering cols

    # splitting
    X = df.drop('Drug', axis=1).values
    y = df['Drug'].values

    X_test, X_train, y_test, y_train = train_test_split(X, y)


    # GaussianNB
    clf = GaussianNB()
    y_pred = train_and_predict(clf, X_test, X_train, y_train)
    analyze_performance("GaussianNB", "Default", y_test, y_pred)

    # Base-DT
    clf = DecisionTreeClassifier()
    y_pred = train_and_predict(clf, X_test, X_train, y_train)
    analyze_performance("Base-DT", "Default", y_test, y_pred, True)

    # Top-DT
    dt = DecisionTreeClassifier()
    clf = GridSearchCV(dt,
        param_grid={
            'criterion' : ['gini', 'entropy'],
            'max_depth' : [5, 8],
            'min_samples_split' : [2, 4, 8]
    },cv=2, return_train_score=False)
    clf.fit(X_train, y_train)

    best_dt_params = clf.best_params_

    clf = DecisionTreeClassifier(criterion=best_mlp_params['criterion'],
                            max_depth=best_mlp_params['max_depth'],
                            min_samples_split=best_mlp_params['min_samples_split'])

    y_pred = train_and_predict(clf, X_test, X_train, y_train)
    analyze_performance("Top-DT", str(best_mlp_params), y_test, y_pred, True)

    # PER
    clf = Perceptron()
    y_pred = train_and_predict(clf, X_test, X_train, y_train)
    analyze_performance("PER", "Default", y_test, y_pred, True)


    # Base-MLP
    clf = MLPClassifier(hidden_layer_sizes=(100), activation='logistic', solver='sgd')
    y_pred = train_and_predict(clf, X_test, X_train, y_train)
    analyze_performance("Base-MLP", """1 hidden layer of 100 neurons, sigmoid/logistic as activation function,
stochastic gradient descent, and default values for the rest of the parameters.""",
        y_test, y_pred, True)


    # Top-MLP
    mlp = MLPClassifier()
    clf = GridSearchCV(mlp,
                       param_grid={
                       'hidden_layer_sizes' : [(30, 50), (10, 10, 10)],
                       'activation' : ['logistic', 'tanh', 'relu', 'identity'],
                       'solver' : ['sgd', 'adam']
                   })


    clf.fit(X_train, y_train)
    best_mlp_params = clf.best_params_

    clf = MLPClassifier(hidden_layer_sizes=best_mlp_params['hidden_layer_sizes'],
                            activation=best_mlp_params['activation'],
                            solver=best_mlp_params['solver'])
    y_pred = train_and_predict(clf, X_test, X_train, y_train)
    analyze_performance("Top-MLP", str(best_mlp_params), y_test, y_pred, True)



    ###############################################################################
    # Redo steps 6 10 times
    ###############################################################################


#     # GaussianNB
    # for i in range(10):

        # clf = GaussianNB()
        # y_pred = train_and_predict(clf, X_test, X_train, y_train)

    # # Base-DT
    # clf = DecisionTreeClassifier()
    # y_pred = train_and_predict(clf, X_test, X_train, y_train)

    # # Top-DT
    # clf = DecisionTreeClassifier(criterion=best_mlp_params['criterion'],
                            # max_depth=best_mlp_params['max_depth'],
                            # min_samples_split=best_mlp_params['min_samples_split'])

    # y_pred = train_and_predict(clf, X_test, X_train, y_train)

    # # PER
    # clf = Perceptron()
    # y_pred = train_and_predict(clf, X_test, X_train, y_train)


    # # Base-MLP
    # clf = MLPClassifier(hidden_layer_sizes=(100), activation='logistic', solver='sgd')
    # y_pred = train_and_predict(clf, X_test, X_train, y_train)



    # # Top-MLP
    # clf = MLPClassifier(hidden_layer_sizes=best_mlp_params['hidden_layer_sizes'],
                            # activation=best_mlp_params['activation'],
                            # solver=best_mlp_params['solver'])
    # y_pred = train_and_predict(clf, X_test, X_train, y_train)



if __name__ == "__main__":
    main()
