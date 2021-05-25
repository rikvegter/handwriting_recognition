import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold

KFOLD = 5

def main():
    #df = pd.read_pickle('local_features.pkl')
    df = pd.read_pickle('pca_components_features.pkl')
    X = df.drop(columns = ['label'])
    y = df['label']


    #Split train test data


    train, test = train_test_split(df, test_size = 0.2)
    X_train, y_train = train.drop(columns = ['label']), train['label']
    X_test, y_test = test.drop(columns = ['label']), test['label']



    #Create classifier
    clf = RandomForestClassifier(random_state=42, n_estimators = 120, criterion = 'entropy')
    clf = svm.SVC(kernel = 'rbf')
    #clf = KNeighborsClassifier(n_neighbors = 3)

    #USE K-fold cross validation to get the accuracy
    kf = KFold(n_splits = KFOLD, shuffle = True, random_state = 42)
    accuracies = cross_val_score(clf, X, y, cv = kf)
    k_fold_accuracy = np.mean(accuracies)
    print('kfold accuracies are ', accuracies)
    print('average accuracy is ', k_fold_accuracy)

if __name__ == "__main__":
    main()
