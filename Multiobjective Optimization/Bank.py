import pandas as pd
import random,time
import numpy as np
import math,copy
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import sklearn.metrics as metrics


from result.measure import calculate_recall,calculate_far,calculate_average_odds_difference, calculate_equal_opportunity_difference, get_counts, measure_final_score
from optimizer.flash import flash_fair_LSR



## Load dataset
dataset_orig = pd.read_csv('dataset/bank.csv')

## Drop categorical features
dataset_orig = dataset_orig.drop(['job', 'marital', 'default',
                     'housing', 'loan', 'contact', 'month', 'day',
                     'poutcome'],axis=1)

## Drop NULL values
dataset_orig = dataset_orig.dropna()
# mean = dataset_orig.loc[:,"age"].mean()
# dataset_orig['age'] = np.where(dataset_orig['age'] >= mean, 1, 0)
dataset_orig['age'] = np.where(dataset_orig['age'] >= 25, 1, 0)
dataset_orig['Probability'] = np.where(dataset_orig['Probability'] == 'yes', 1, 0)


## Chaneg symbolic to numeric column
from sklearn.preprocessing import LabelEncoder
gle = LabelEncoder()
genre_labels = gle.fit_transform(dataset_orig['education'])
genre_mappings = {index: label for index, label in enumerate(gle.classes_)}
dataset_orig['education'] = genre_labels


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
dataset_orig = pd.DataFrame(scaler.fit_transform(dataset_orig),columns = dataset_orig.columns)

def run_ten_times_default():
    print(" ---------- Default Results --------")
    for i in range(3):
        print("----Run No----",i)
        start = time.time()
        ## Divide into train,validation,test
        dataset_orig_train, dataset_orig_vt = train_test_split(dataset_orig, test_size=0.3)
        dataset_orig_valid, dataset_orig_test = train_test_split(dataset_orig_vt, test_size=0.5)

        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'], dataset_orig_train['Probability']
        X_valid, y_valid = dataset_orig_valid.loc[:, dataset_orig_valid.columns != 'Probability'], dataset_orig_valid['Probability']
        X_test, y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'], dataset_orig_test['Probability']

        #### DEFAULT Learners ####
        # --- LSR
        clf = LogisticRegression(C=1.0, penalty='l2', solver='liblinear', max_iter=100) # LSR Default Config
        # --- CART
        # clf = tree.DecisionTreeClassifier(criterion="gini",splitter="best",min_samples_leaf=1,min_samples_split=2) # CART Default Config
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        cnf_matrix_test = confusion_matrix(y_test, y_pred)

        print(cnf_matrix_test)

        TN, FP, FN, TP = confusion_matrix(y_test,y_pred).ravel()

        print("recall:", 1 - calculate_recall(TP,FP,FN,TN))
        print("far:",calculate_far(TP,FP,FN,TN))
        print("aod for age:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'age', 'aod'))
        print("eod for age:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'age', 'eod'))        
        end = time.time()
        print(end - start)

def run_ten_times_FLASH():
    print(" ---------- FLASH Results --------")
    for i in range(3):
        print("----Run No----",i)
        start = time.time()
        ## Divide into train,validation,test
        dataset_orig_train, dataset_orig_vt = train_test_split(dataset_orig, test_size=0.3)
        dataset_orig_valid, dataset_orig_test = train_test_split(dataset_orig_vt, test_size=0.5)

        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'], dataset_orig_train['Probability']
        X_valid, y_valid = dataset_orig_valid.loc[:, dataset_orig_valid.columns != 'Probability'], dataset_orig_valid['Probability']
        X_test, y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'], dataset_orig_test['Probability']


        # tuner = LR_TUNER()
        # best_config = tune_with_flash(tuner,  X_train, y_train, X_valid, y_valid, 'adult', dataset_orig_valid, 'sex')
        best_config = flash_fair_LSR(dataset_orig,"age","ABCD")
        print("best_config",best_config)
        p1 = best_config[0]
        if best_config[1] == 1:
            p2 = 'l1'
        else:
            p2 = 'l2'
        if best_config[2] == 1:
            p3 = 'liblinear'
        else:
            p3 = 'saga'
        p4 = best_config[3]
        clf = LogisticRegression(C=p1, penalty=p2, solver=p3, max_iter=p4)
        # clf = tuner.get_clf(best_config)
        print("recall :", 1 - measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'age', 'recall'))
        print("far :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'age', 'far'))
        print("aod :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'age', 'aod'))
        print("eod :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'age', 'eod'))
        end = time.time()
        print(end - start)


# run_ten_times_default()
run_ten_times_FLASH()



