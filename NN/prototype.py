from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import idx2numpy

X_train = idx2numpy.convert_from_file("data/train-images-idx3-ubyte")
Y_train = idx2numpy.convert_from_file("data/train-labels-idx1-ubyte")
X_test = idx2numpy.convert_from_file("data/t10k-images-idx3-ubyte")
Y_test = idx2numpy.convert_from_file("data/t10k-labels-idx1-ubyte")

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2])
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1]*X_test.shape[2])
# Normalize the data
MinMaxScaler = preprocessing.MinMaxScaler()
X_data_minmax = MinMaxScaler.fit_transform(X_train)
X_test_data_minmax = MinMaxScaler.fit_transform(X_test)

def subsample_training(X, Y, M):
    from numpy.random import default_rng
    assert X.shape[0] == Y.shape[0]
    assert M < X.shape[0]
    rng = default_rng()
    idxs = rng.choice(X.shape[0], size=M, replace=False)
    return X.take(idxs, axis=0), Y.take(idxs, axis=0)

def remove_outliers(X, Y):
    import ipdb;ipdb.set_trace()
    knn_clf=KNeighborsClassifier(n_neighbors=5, metric="euclidean")
    knn_clf.fit(X, Y)
    ypred=knn_clf.predict(X)
    outliers = ypred != Y
    # remove outliers
    X, Y = X[~outliers], Y[~outliers]
    return X, Y


def keep_useful(X, Y):
    useful_X, useful_Y = [], []
    labels_vis = set()
    for x, y in zip(X, Y):
        if y not in labels_vis:
            useful_X.append(x)
            useful_Y.append(y)
            labels_vis.add(y)
        else:
            # Check if the value is misclassified
            import ipdb;ipdb.set_trace()
            knn_clf=KNeighborsClassifier(n_neighbors=1, metric="euclidean")
            knn_clf.fit(useful_X, useful_Y)
            ypred=knn_clf.predict([x])
            if ypred[0] != y:
                useful_X.append(x)
                useful_Y.append(y)
    return useful_X, useful_Y            


def prototype_selection(X, Y):
    X, Y = remove_outliers(X, Y)
    X, Y = keep_useful(X, Y)
    return X, Y    

def predict_values_and_calculate_accuracy_uniform_random(M):
    # 1 NN, distance = Euclidian
    knn_clf=KNeighborsClassifier(n_neighbors=1, metric="euclidean")
    X, Y = subsample_training(X_data_minmax, Y_train, M)
    knn_clf.fit(X, Y)
    ypred=knn_clf.predict(X_test_data_minmax)
    result = confusion_matrix(Y_test, ypred)
    print("Confusion Matrix:")
    print(result)
    result1 = classification_report(Y_test, ypred)
    print("Classification Report:",)
    print (result1)
    result2 = accuracy_score(Y_test,ypred)
    # print(type(result1))
    print("Accuracy:",result2)
    return result2

def predict_values_and_calculate_accuracy_prototype_subset(M):
    # 1 NN, distance = Euclidian
    knn_clf=KNeighborsClassifier(n_neighbors=1, metric="euclidean")
    X, Y = subsample_training(X_data_minmax, Y_train, M)
    X, Y = prototype_selection(X, Y)

    knn_clf.fit(X, Y)
    ypred=knn_clf.predict(X_test_data_minmax)
    # result = confusion_matrix(Y_test, ypred)
    # print("Confusion Matrix:")
    # print(result)
    #result1 = classification_report(Y_test, ypred)
    #print("Classification Report:",)
    #print (result1)
    result2 = accuracy_score(Y_test,ypred)
    #print("Accuracy:",result2)
    return result2


def test_multiple_iterations(num_iterations, M):
    accuracies_uniform = []
    accuracies_prototype_selection = []
    for i in range(num_iterations):
        accuracies_uniform.append(predict_values_and_calculate_accuracy_uniform_random(M))
        accuracies_prototype_selection.append(predict_values_and_calculate_accuracy_prototype_subset(M))
    print(accuracies_uniform)    
    import matplotlib.pyplot as plt
    import numpy as np

    #plt.ylim(-0.2, 2)
    labels = [f"{M}_{i}" for i in range(num_iterations)]

    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, accuracies_uniform, width, label='Men')
    rects2 = ax.bar(x + width/2, accuracies_uniform, width, label='Women')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylim(-0.2, 2)
    ax.set_ylabel('Scores')
    ax.set_title('Scores by group and gender')
    #ax.set_xticks(x, labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)

    fig.tight_layout()

    plt.show()    


test_multiple_iterations(5, 1000)