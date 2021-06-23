import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold
import pickle
KFOLD = 5

DROP_PC_COMPONENTS_UNTIL = 25

def drop_column_names(drop_pc_components_from, drop_pc_components_until):
    components = []
    for i in range(drop_pc_components_from, drop_pc_components_until):
        component = 'pc' + str(i)
        components.append(component)
    return components

def main(pc_components):
    df = pd.read_pickle('local_features.pkl')
    #df = pd.read_pickle('pca_components_features.pkl')

    columns_to_drop = drop_column_names(pc_components + 1, DROP_PC_COMPONENTS_UNTIL + 1)
    df = df.drop(columns=columns_to_drop)
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
    clf = RandomForestClassifier(random_state=42, n_estimators = 120, criterion = 'entropy')
    #clf = svm.SVC(kernel = 'rbf')
    #clf = KNeighborsClassifier(n_neighbors = 5)

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

    #return kf-accuracy
    train_acc = accuracy_score(y_true, y_pred)
    test_acc = k_fold_accuracy

    pkl_filename = 'pickle_model.pkl'
    with open(pkl_filename, 'wb') as file:
        pickle.dump(clf, file)

    return train_acc, test_acc
if __name__ == "__main__":

    pc_components = 33
    test_accuracies = []
    train_accuracies = []
    main(25)
    '''
    for i in range(2,pc_components):
        train_acc, test_acc = main(i)
        test_accuracies.append([test_acc, i])
        train_accuracies.append([train_acc, i])
    '''
    import pdb; pdb.set_trace()


    #main()
