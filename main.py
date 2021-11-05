import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from data_prepare import get_prepared_data

if __name__ == '__main__':
    csv_file = 'csv/HR.csv'
    df = pd.read_csv(csv_file)
    X = pd.read_csv(csv_file, usecols=lambda col: col not in ['Attrition'])
    y = pd.read_csv(csv_file, usecols=['Attrition'])
    # targets preparation
    y = y.replace("No",0)
    y = y.replace('Yes',1)
    y = np.array(y).reshape((y.shape[0],))

    # dataset preparation
    X = get_prepared_data(X)
    # print(X.head())
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    clf = LogisticRegression(C=100,max_iter=100000).fit(X_train, y_train)
    print("Accuracy on train data:", clf.score(X_train, y_train))
    print("Accuracy on test data:", clf.score(X_test, y_test))
    print("Number of attributes:",len(clf.coef_[0]))
