from collections import OrderedDict
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn import neighbors

import sklearn

import warnings
import flwr as fl

import pandas as pd
import os
import sys
import numpy as np


def load_data():
    data = np.load('data.npz')
    X = data['x']
    y = data['y']

    percentage_split = 0.35

    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=percentage_split, random_state=42, shuffle=True)
    num_examples = {"trainset": len(X_train), "testset": len(X_test)}

    return X_train, X_test, Y_train, Y_test, num_examples

def train(model, X_train, Y_train):
    print(f"Training....")
    model.fit(X_train, Y_train)
    print(f"Trained.")

def test(model, X_test, Y_test):
    print(f"Testing....")
    predictions = model.predict(X_test)
    print(f"Tested")

    # test_length = len(Y_test)
    loss = 0.0

    accuracy = metrics.accuracy_score(Y_test, predictions)
    f1_score = metrics.f1_score(Y_test, predictions, average='macro')
    precision = metrics.precision_score(Y_test, predictions, average='macro')
    recall = metrics.recall_score(Y_test, predictions, average='macro')

    # tn, fp, fn, tp = metrics.confusion_matrix(Y_test, predictions).ravel()
    # print(f"Number of test samples: {test_length}")
    print(f"Testing accuracy: { accuracy * 100.0 }% ")
    print(f"F1 score: {f1_score}")
    print(f"Precision {precision*100.0 }% ")
    print(f"Recall : {recall * 100.0}%")
	# print(f"-=-=-=-=-=-=-=-=-=-=-")
	# # print(f"True positives/malicious: {tp} out of {tp + fp + fn + tn} ({ (tp / test_length) * 100.0 }%)")
	# print(f"False positives/malicious: {fp} out of {tp + fp + fn + tn} ({ (fp / test_length) * 100.0 }%)")
	# print(f"True negatives/benign: {tn} out of {tp + fp + fn + tn} ({ (tn / test_length) * 100.0 }%)")
	# print(f"False negatives/benign: {fn} out of {tp + fp + fn + tn} ({ (fn / test_length) * 100.0 }%)")
    
    return loss, accuracy

def set_initial_params(model):
    #Logestic Regression
    # n_classes = 7
    # n_features = 4
    # model.classes_ = np.array([i for i in range(7)])

    # model.coef_ = np.zeros((n_classes, n_features))
    # if model.fit_intercept:
    #     model.intercept_ = np.zeros((n_classes,))

    #SVM
    n_classes = 7  
    n_features = 4  # Number of features in dataset
    model.offset_=np.zeros(n_classes)
    model.coef_ = np.zeros((n_classes, n_features))


def main():

    # model = LogisticRegression(
    #     penalty="l2",
    #     max_iter=1,  # local epoch
    #     warm_start=True,  # prevent refreshing weights when fitting
    # )
    #model=svm.LinearSVC(C=0.82, max_iter=1_000, tol=0.0001, dual=False)
    model = RandomForestClassifier(n_estimators=40, max_depth=5)

    #Load Data
    X_train, X_test, Y_train, Y_test, num_examples = load_data()

    # model = neighbors.KNeighborsClassifier(algorithm='brute', n_neighbors=3, metric='euclidean')

    set_initial_params(model)

    #Flower Client

    class DisClient(fl.client.NumPyClient):
        def get_parameters(self, config):
            #return local model parameters
            #return [model.get_params()]
            # return [model.coef_, model.intercept_]
            return [model.coef_, model.offset_]
        
        def set_parameters(self, parameters):
            model.coef_ = parameters[0]
            #LR below code
            # if model.fit_intercept:
            model.offset_ = parameters[1]
            # for parameter in parameters[0]:
            #     st_param = str(parameter)
            #     val = model.get_params()[st_param]
            #     model.set_params(**{st_param: val})

            # model.coef_ = parameters[0]
            # model.intercept_ = parameters[1]

        def fit(self, parameters, config):
            model.set_params
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model.fit(X_train, Y_train)
            return self.get_parameters(config), len(X_train), {}
            # try:
            #     self.set_parameters(parameters)
            #     with warnings.catch_warnings():
            #         warnings.simplefilter("ignore")
            #         train(model, X_train, Y_train)
            # except:
            #     print("Nothing")

            #     return self.get_parameters(), num_examples["trainset"]
            # return self.get_parameters(), num_examples["trainset"]

        
        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss, accuracy = test(model, X_test, Y_test)
            return loss, num_examples["testset"], {"accuracy": accuracy}

#start Client
    fl.client.start_numpy_client(server_address="localhost:8080", client=DisClient())
    #train(model, X_train, Y_train)
    #test(model, X_test, Y_test)

if __name__ == "__main__":
    main()






