import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold

KFOLD = 5
DROP_PC_COMPONENTS_FROM = 20

def drop_column_names(drop_pc_components_from):
    components = []
    for i in range(drop_pc_components_from, 21):
        component = 'pc' + str(i)
        components.append(component)
    return components

def main():
    #df = pd.read_pickle('local_features.pkl')
    df = pd.read_pickle('pca_components_features.pkl')


    #columns_to_drop = drop_column_names(DROP_PC_COMPONENTS_FROM)
    #df = df.drop(columns=columns_to_drop)

    X = df.drop(columns = ['label'])
    y = df['label']


    #Split train test data
    train, test = train_test_split(df, test_size = 0.2)
    X_train, y_train = train.drop(columns = ['label']), train['label']
    X_test, y_test = test.drop(columns = ['label']), test['label']


    #####################################################$#
    ########## Most optimal setting four so far ###########
    ########### Random Forest with 120 trees ##############
    #####################################################$#

    #Create classifier
    clf = RandomForestClassifier(random_state=42, n_estimators = 10, criterion = 'entropy')
    #clf = svm.SVC(kernel = 'rbf')
    #clf = KNeighborsClassifier(n_neighbors = 7)

    #USE K-fold cross validation to get the accuracy
    kf = KFold(n_splits = KFOLD, shuffle = True, random_state = 42)

    accuracies = cross_val_score(clf, X, y, cv = kf)
    k_fold_accuracy = np.mean(accuracies)
    print('kfold accuracies are ', accuracies)
    print('average test accuracy is ', k_fold_accuracy)

    print('-------------Check for overfitting-------------')
    clf.fit(X_train, y_train)
    y_true = y_train
    y_pred = clf.predict(X_train)
    print('training accuracy = ', accuracy_score(y_true, y_pred))
if __name__ == "__main__":
    main()
