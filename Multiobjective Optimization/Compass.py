import pandas as pd
import numpy as np
import random,time
import math,copy
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn import tree
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier


from result.measure import calculate_recall,calculate_far,calculate_average_odds_difference, calculate_equal_opportunity_difference, get_counts, measure_final_score
from optimizer.flash import flash_fair_LSR


## Load dataset
dataset_orig = pd.read_csv('dataset/compas-scores-two-years.csv')



## Drop categorical features
## Removed two duplicate coumns - 'decile_score','priors_count'
dataset_orig = dataset_orig.drop(['id','name','first','last','compas_screening_date','dob','age','juv_fel_count','decile_score','juv_misd_count','juv_other_count','days_b_screening_arrest','c_jail_in','c_jail_out','c_case_number','c_offense_date','c_arrest_date','c_days_from_compas','c_charge_desc','is_recid','r_case_number','r_charge_degree','r_days_from_arrest','r_offense_date','r_charge_desc','r_jail_in','r_jail_out','violent_recid','is_violent_recid','vr_case_number','vr_charge_degree','vr_offense_date','vr_charge_desc','type_of_assessment','decile_score','score_text','screening_date','v_type_of_assessment','v_decile_score','v_score_text','v_screening_date','in_custody','out_custody','start','end','event'],axis=1)

## Drop NULL values
dataset_orig = dataset_orig.dropna()


## Change symbolics to numerics
dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'Female', 1, 0)
dataset_orig['race'] = np.where(dataset_orig['race'] != 'Caucasian', 0, 1)
dataset_orig['priors_count'] = np.where((dataset_orig['priors_count'] >= 1 ) & (dataset_orig['priors_count'] <= 3), 3, dataset_orig['priors_count'])
dataset_orig['priors_count'] = np.where(dataset_orig['priors_count'] > 3, 4, dataset_orig['priors_count'])
dataset_orig['age_cat'] = np.where(dataset_orig['age_cat'] == 'Greater than 45',45,dataset_orig['age_cat'])
dataset_orig['age_cat'] = np.where(dataset_orig['age_cat'] == '25 - 45', 25, dataset_orig['age_cat'])
dataset_orig['age_cat'] = np.where(dataset_orig['age_cat'] == 'Less than 25', 0, dataset_orig['age_cat'])
dataset_orig['c_charge_degree'] = np.where(dataset_orig['c_charge_degree'] == 'F', 1, 0)
## Rename class column
dataset_orig.rename(index=str, columns={"two_year_recid": "Probability"}, inplace=True)




## Divide into train,validation,test
dataset_orig_train, dataset_orig_vt = train_test_split(dataset_orig, test_size=0.3, random_state=0)
dataset_orig_valid, dataset_orig_test = train_test_split(dataset_orig_vt, test_size=0.5, random_state=0)


X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'], dataset_orig_train['Probability']
X_valid , y_valid = dataset_orig_valid.loc[:, dataset_orig_valid.columns != 'Probability'], dataset_orig_valid['Probability']
X_test , y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'], dataset_orig_test['Probability']


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
        print("aod for sex:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'sex', 'aod'))
        print("eod for sex:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'sex', 'eod'))
        print("aod for race:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'race', 'aod'))
        print("eod for race:",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'race', 'eod'))
        end = time.time()
        print(end-start)

def run_ten_times_FLASH():
    print(" ---------- FLASH Results --------")
    for i in range(3):
        print("----Run No----",i)
        start = time.time()
        ## Divide into train,validation,test
        dataset_orig_train, dataset_orig_vt = train_test_split(dataset_orig,  test_size=0.3)
        dataset_orig_valid, dataset_orig_test = train_test_split(dataset_orig_vt,  test_size=0.5)

        X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'], dataset_orig_train['Probability']
        X_valid, y_valid = dataset_orig_valid.loc[:, dataset_orig_valid.columns != 'Probability'], dataset_orig_valid['Probability']
        X_test, y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'], dataset_orig_test['Probability']


        # tuner = LR_TUNER()
        # best_config = tune_with_flash(tuner,  X_train, y_train, X_valid, y_valid, 'adult', dataset_orig_valid, 'sex')
        best_config = flash_fair_LSR(dataset_orig,"sex","ABCD")
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
        print("recall :", 1 - measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'sex', 'recall'))
        print("far :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'sex', 'far'))
        print("aod :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'sex', 'aod'))
        print("eod :",measure_final_score(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'sex', 'eod'))
        end = time.time()
        print(end - start)


run_ten_times_default()
run_ten_times_FLASH()


