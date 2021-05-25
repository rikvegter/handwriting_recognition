import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

def remove_feature_with_low_variance(df):

    return df

def main():
    df = pd.read_pickle('local_features.pkl')
    df = df.drop(columns = ['height'])

    #Split train test data
    train, test = train_test_split(df, test_size = 0.2)
    X_train, y_train = train.drop(columns = ['label']), train['label']
    X_test, y_test = test.drop(columns = ['label']), test['label']



    #Train SVM
    clf = RandomForestClassifier(random_state=42, n_estimators = 60)
    #clf = KNeighborsClassifier(n_neighbors = 9)

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('accuracy = ', accuracy * 100, '%')

if __name__ == "__main__":
    main()
