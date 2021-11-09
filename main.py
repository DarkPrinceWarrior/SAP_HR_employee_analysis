import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from data_prepare import get_prepared_data


def get_major_attributes(weights, X):

    column_names = X.columns

    for index,w in enumerate(weights):
        if w !=0:
            print(column_names[index])


if __name__ == '__main__':
    csv_file = 'csv/HR.csv'
    X = pd.read_csv(csv_file, usecols=lambda col: col not in ['Attrition'])
    y = pd.read_csv(csv_file, usecols=['Attrition'])
    # targets preparation
    y = y.replace("No", 0)
    y = y.replace('Yes', 1)
    y = np.array(y).reshape((y.shape[0],))

    # dataset preparation
    X = get_prepared_data(X)
    # print(X.head())
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    # clf = LogisticRegression(C=0.0001, max_iter=100000).fit(X_train, y_train)
    clf = LogisticRegression(C=0.01, penalty='l1', max_iter=100000, solver='liblinear').fit(X_train, y_train)
    print("Accuracy on train data:", clf.score(X_train, y_train))
    print("Accuracy on test data:", clf.score(X_test, y_test))
    weights_array = clf.coef_
    print("Number of attributes:", len(weights_array[0]))
    print("weights coefs values :\n", weights_array)
    print("---------------------------------------------------------")
    # lets reshape this coefs array into 1 dimensional array = "list"
    weights_list=list(weights_array[0])
    # print(weights_list)
    get_major_attributes(weights_list,X)