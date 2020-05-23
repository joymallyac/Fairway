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
from sklearn.svm import SVC

from result.measure import calculate_average_odds_difference, calculate_equal_opportunity_difference, get_counts, measure_final_score_german


## Load dataset
dataset_orig = pd.read_csv('dataset/GermanData.csv')

## Drop categorical features
dataset_orig = dataset_orig.drop(['1','2','4','5','8','10','11','12','14','15','16','17','18','19','20'],axis=1)

## Drop NULL values
dataset_orig = dataset_orig.dropna()


## Change symbolics to numerics
dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'A91', 1, dataset_orig['sex'])
dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'A92', 0, dataset_orig['sex'])
dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'A93', 1, dataset_orig['sex'])
dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'A94', 1, dataset_orig['sex'])
dataset_orig['sex'] = np.where(dataset_orig['sex'] == 'A95', 0, dataset_orig['sex'])
dataset_orig['age'] = np.where(dataset_orig['age'] >= 25, 1, 0)
dataset_orig['credit_history'] = np.where(dataset_orig['credit_history'] == 'A30', 1, dataset_orig['credit_history'])
dataset_orig['credit_history'] = np.where(dataset_orig['credit_history'] == 'A31', 1, dataset_orig['credit_history'])
dataset_orig['credit_history'] = np.where(dataset_orig['credit_history'] == 'A32', 1, dataset_orig['credit_history'])
dataset_orig['credit_history'] = np.where(dataset_orig['credit_history'] == 'A33', 2, dataset_orig['credit_history'])
dataset_orig['credit_history'] = np.where(dataset_orig['credit_history'] == 'A34', 3, dataset_orig['credit_history'])

dataset_orig['savings'] = np.where(dataset_orig['savings'] == 'A61', 1, dataset_orig['savings'])
dataset_orig['savings'] = np.where(dataset_orig['savings'] == 'A62', 1, dataset_orig['savings'])
dataset_orig['savings'] = np.where(dataset_orig['savings'] == 'A63', 2, dataset_orig['savings'])
dataset_orig['savings'] = np.where(dataset_orig['savings'] == 'A64', 2, dataset_orig['savings'])
dataset_orig['savings'] = np.where(dataset_orig['savings'] == 'A65', 3, dataset_orig['savings'])

dataset_orig['employment'] = np.where(dataset_orig['employment'] == 'A72', 1, dataset_orig['employment'])
dataset_orig['employment'] = np.where(dataset_orig['employment'] == 'A73', 1, dataset_orig['employment'])
dataset_orig['employment'] = np.where(dataset_orig['employment'] == 'A74', 2, dataset_orig['employment'])
dataset_orig['employment'] = np.where(dataset_orig['employment'] == 'A75', 2, dataset_orig['employment'])
dataset_orig['employment'] = np.where(dataset_orig['employment'] == 'A71', 3, dataset_orig['employment'])



## ADD Columns
dataset_orig['credit_history=Delay'] = 0
dataset_orig['credit_history=None/Paid'] = 0
dataset_orig['credit_history=Other'] = 0

dataset_orig['credit_history=Delay'] = np.where(dataset_orig['credit_history'] == 1, 1, dataset_orig['credit_history=Delay'])
dataset_orig['credit_history=None/Paid'] = np.where(dataset_orig['credit_history'] == 2, 1, dataset_orig['credit_history=None/Paid'])
dataset_orig['credit_history=Other'] = np.where(dataset_orig['credit_history'] == 3, 1, dataset_orig['credit_history=Other'])

dataset_orig['savings=500+'] = 0
dataset_orig['savings=<500'] = 0
dataset_orig['savings=Unknown/None'] = 0

dataset_orig['savings=500+'] = np.where(dataset_orig['savings'] == 1, 1, dataset_orig['savings=500+'])
dataset_orig['savings=<500'] = np.where(dataset_orig['savings'] == 2, 1, dataset_orig['savings=<500'])
dataset_orig['savings=Unknown/None'] = np.where(dataset_orig['savings'] == 3, 1, dataset_orig['savings=Unknown/None'])

dataset_orig['employment=1-4 years'] = 0
dataset_orig['employment=4+ years'] = 0
dataset_orig['employment=Unemployed'] = 0

dataset_orig['employment=1-4 years'] = np.where(dataset_orig['employment'] == 1, 1, dataset_orig['employment=1-4 years'])
dataset_orig['employment=4+ years'] = np.where(dataset_orig['employment'] == 2, 1, dataset_orig['employment=4+ years'])
dataset_orig['employment=Unemployed'] = np.where(dataset_orig['employment'] == 3, 1, dataset_orig['employment=Unemployed'])


dataset_orig = dataset_orig.drop(['credit_history','savings','employment'],axis=1)

# dataset_orig['Probability'] = np.where(dataset_orig['Probability'] == 2, 0, 1)

def calculate_recall(TP,FP,FN,TN):
    if (TP + FN) != 0:
        recall = TP / (TP + FN)
    # To maximize recall, we will return ( 1 - recall)
    return 1 - recall

def calculate_far(TP,FP,FN,TN):
    if (FP + TN) != 0:
        far = FP / (FP + TN)
    else:
        far = 0
    return far

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

        print("recall:", 1 - measure_final_score_german(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'sex', 'recall'))
        print("far:",measure_final_score_german(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'sex', 'far'))
        print("aod for sex:",measure_final_score_german(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'sex', 'aod'))
        print("eod for sex:",measure_final_score_german(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'sex', 'eod'))
        print("aod for age:",measure_final_score_german(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'age', 'aod'))
        print("eod for age:",measure_final_score_german(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'age', 'eod'))
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
        # best_config = tune_with_flash(tuner,  X_train, y_train, X_valid, y_valid, 'german', dataset_orig_valid, 'sex')
        # print("best_config",best_config)
        # clf = tuner.get_clf(best_config)
        best_config = flash_fair_LSR("sex", "ABCD")
        print("best_config", best_config)
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
        print("recall for sex:", 1 - measure_final_score_german(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'sex', 'recall'))
        print("far for sex:",measure_final_score_german(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'sex', 'far'))
        print("aod for sex:",measure_final_score_german(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'sex', 'aod'))
        print("eod for sex:",measure_final_score_german(dataset_orig_test, clf, X_train, y_train, X_test, y_test, 'sex', 'eod'))
        end = time.time()
        print(end - start)
		

def measure_scores(X_train, y_train, X_valid, y_valid, valid_df, biased_col, clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_valid)
    cnf_matrix = confusion_matrix(y_valid, y_pred)

    TN, FP, FN, TP = confusion_matrix(y_valid,y_pred).ravel()

    valid_df_copy = copy.deepcopy(valid_df)
    valid_df_copy['current_pred_' + biased_col] = y_pred

    valid_df_copy['TP_' + biased_col + "_1"] = np.where((valid_df_copy['Probability'] == 1) &
                                                        (valid_df_copy[
                                                             'current_pred_' + biased_col] == 1) &
                                                        (valid_df_copy[biased_col] == 1), 1, 0)

    valid_df_copy['TN_' + biased_col + "_1"] = np.where((valid_df_copy['Probability'] == 2) &
                                                        (valid_df_copy[
                                                             'current_pred_' + biased_col] == 2) &
                                                        (valid_df_copy[biased_col] == 1), 1, 0)

    valid_df_copy['FN_' + biased_col + "_1"] = np.where((valid_df_copy['Probability'] == 1) &
                                                        (valid_df_copy[
                                                             'current_pred_' + biased_col] == 2) &
                                                        (valid_df_copy[biased_col] == 1), 1, 0)

    valid_df_copy['FP_' + biased_col + "_1"] = np.where((valid_df_copy['Probability'] == 2) &
                                                        (valid_df_copy[
                                                             'current_pred_' + biased_col] == 1) &
                                                        (valid_df_copy[biased_col] == 1), 1, 0)

    valid_df_copy['TP_' + biased_col + "_0"] = np.where((valid_df_copy['Probability'] == 1) &
                                                        (valid_df_copy[
                                                             'current_pred_' + biased_col] == 1) &
                                                        (valid_df_copy[biased_col] == 0), 1, 0)

    valid_df_copy['TN_' + biased_col + "_0"] = np.where((valid_df_copy['Probability'] == 2) &
                                                        (valid_df_copy[
                                                             'current_pred_' + biased_col] == 2) &
                                                        (valid_df_copy[biased_col] == 0), 1, 0)

    valid_df_copy['FN_' + biased_col + "_0"] = np.where((valid_df_copy['Probability'] == 1) &
                                                        (valid_df_copy[
                                                             'current_pred_' + biased_col] == 2) &
                                                        (valid_df_copy[biased_col] == 0), 1, 0)

    valid_df_copy['FP_' + biased_col + "_0"] = np.where((valid_df_copy['Probability'] == 2) &
                                                        (valid_df_copy[
                                                             'current_pred_' + biased_col] == 1) &
                                                        (valid_df_copy[biased_col] == 0), 1, 0)

    a = valid_df_copy['TP_' + biased_col + "_1"].sum()
    b = valid_df_copy['TN_' + biased_col + "_1"].sum()
    c = valid_df_copy['FN_' + biased_col + "_1"].sum()
    d = valid_df_copy['FP_' + biased_col + "_1"].sum()
    e = valid_df_copy['TP_' + biased_col + "_0"].sum()
    f = valid_df_copy['TN_' + biased_col + "_0"].sum()
    g = valid_df_copy['FN_' + biased_col + "_0"].sum()
    h = valid_df_copy['FP_' + biased_col + "_0"].sum()

    recall = calculate_recall(TP, FP, FN, TN)
    far = calculate_far(TP, FP, FN, TN)
    aod = calculate_average_odds_difference(a, b, c, d, e, f, g, h)
    eod = calculate_equal_opportunity_difference(a, b, c, d, e, f, g, h)

    return recall, far, aod, eod


def flash_fair_LSR(biased_col, n_obj):  # biased_col can be "sex" or "race", n_obj can be "ABCD" or "AB" or "CD"

    dataset_orig_train, dataset_orig_vt = train_test_split(dataset_orig, test_size=0.3)
    dataset_orig_valid, dataset_orig_test = train_test_split(dataset_orig_vt, test_size=0.5)

    X_train, y_train = dataset_orig_train.loc[:, dataset_orig_train.columns != 'Probability'], dataset_orig_train[
        'Probability']
    X_valid, y_valid = dataset_orig_valid.loc[:, dataset_orig_valid.columns != 'Probability'], dataset_orig_valid[
        'Probability']
    X_test, y_test = dataset_orig_test.loc[:, dataset_orig_test.columns != 'Probability'], dataset_orig_test[
        'Probability']

    def convert_lsr(index):  # 30 2 2 100
        a = int(index / 400 + 1)
        b = int(index % 400 / 200 + 1)
        c = int(index % 200 / 100 + 1)
        d = int(index % 100 + 10)
        return a, b, c, d

    all_case = set(range(0, 12000))
    modeling_pool = random.sample(all_case, 20)

    List_X = []
    List_Y = []

    for i in range(len(modeling_pool)):
        temp = convert_lsr(modeling_pool[i])
        List_X.append(temp)
        p1 = temp[0]
        if temp[1] == 1:
            p2 = 'l1'
        else:
            p2 = 'l2'
        if temp[2] == 1:
            p3 = 'liblinear'
        else:
            p3 = 'saga'
        p4 = temp[3]
        model = LogisticRegression(C=p1, penalty=p2, solver=p3, max_iter=p4)

        all_value = measure_scores(X_train, y_train, X_valid, y_valid, dataset_orig_valid, biased_col, model)
        four_goal = all_value[0] + all_value[1] + all_value[2] + all_value[3]
        two_goal_recall_far = all_value[0] + all_value[1]
        two_goal_aod_eod = all_value[2] + all_value[3]
        if n_obj == "ABCD":
            List_Y.append(four_goal)
        elif n_obj == "AB":
            List_Y.append(two_goal_recall_far)
        elif n_obj == "CD":
            List_Y.append(two_goal_aod_eod)
        else:
            print("Wrong number of objects")

    remain_pool = all_case - set(modeling_pool)
    test_list = []
    for i in list(remain_pool):
        test_list.append(convert_lsr(i))

    upper_model = DecisionTreeRegressor()
    life = 20

    while len(List_X) < 200 and life > 0:
        upper_model.fit(List_X, List_Y)
        candidate = random.sample(test_list, 1)
        test_list.remove(candidate[0])
        candi_pred_value = upper_model.predict(candidate)
        if candi_pred_value < np.median(List_Y):
            List_X.append(candidate[0])
            candi_config = candidate[0]

            pp1 = candi_config[0]
            if candi_config[1] == 1:
                pp2 = 'l1'
            else:
                pp2 = 'l2'
            if candi_config[2] == 1:
                pp3 = 'liblinear'
            else:
                pp3 = 'saga'
            pp4 = candi_config[3]

            candi_model = LogisticRegression(C=pp1, penalty=pp2, solver=pp3, max_iter=pp4)
            candi_value = measure_scores(X_train, y_train, X_valid, y_valid, dataset_orig_valid, biased_col,
                                         candi_model)
            candi_four_goal = candi_value[0] + candi_value[1] + candi_value[2] + candi_value[3]
            candi_two_goal_recall_far = candi_value[0] + candi_value[1]
            candi_two_goal_aod_eod = candi_value[2] + candi_value[3]
            if n_obj == "ABCD":
                List_Y.append(candi_four_goal)
            elif n_obj == "AB":
                List_Y.append(candi_two_goal_recall_far)
            elif n_obj == "CD":
                List_Y.append(candi_two_goal_aod_eod)
        else:
            life -= 1

    min_index = int(np.argmin(List_Y))

    return List_X[min_index]


# run_ten_times_default()
run_ten_times_FLASH()




