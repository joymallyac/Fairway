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


from result.measure import calculate_recall,calculate_far,calculate_average_odds_difference, calculate_equal_opportunity_difference, get_counts, measure_final_score


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

    valid_df_copy['TN_' + biased_col + "_1"] = np.where((valid_df_copy['Probability'] == 0) &
                                                        (valid_df_copy[
                                                             'current_pred_' + biased_col] == 0) &
                                                        (valid_df_copy[biased_col] == 1), 1, 0)

    valid_df_copy['FN_' + biased_col + "_1"] = np.where((valid_df_copy['Probability'] == 1) &
                                                        (valid_df_copy[
                                                             'current_pred_' + biased_col] == 0) &
                                                        (valid_df_copy[biased_col] == 1), 1, 0)

    valid_df_copy['FP_' + biased_col + "_1"] = np.where((valid_df_copy['Probability'] == 0) &
                                                        (valid_df_copy[
                                                             'current_pred_' + biased_col] == 1) &
                                                        (valid_df_copy[biased_col] == 1), 1, 0)

    valid_df_copy['TP_' + biased_col + "_0"] = np.where((valid_df_copy['Probability'] == 1) &
                                                        (valid_df_copy[
                                                             'current_pred_' + biased_col] == 1) &
                                                        (valid_df_copy[biased_col] == 0), 1, 0)

    valid_df_copy['TN_' + biased_col + "_0"] = np.where((valid_df_copy['Probability'] == 0) &
                                                        (valid_df_copy[
                                                             'current_pred_' + biased_col] == 0) &
                                                        (valid_df_copy[biased_col] == 0), 1, 0)

    valid_df_copy['FN_' + biased_col + "_0"] = np.where((valid_df_copy['Probability'] == 1) &
                                                        (valid_df_copy[
                                                             'current_pred_' + biased_col] == 0) &
                                                        (valid_df_copy[biased_col] == 0), 1, 0)

    valid_df_copy['FP_' + biased_col + "_0"] = np.where((valid_df_copy['Probability'] == 0) &
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


def flash_fair_LSR(dataset_orig, biased_col, n_obj):  # biased_col can be "sex" or "race", n_obj can be "ABCD" or "AB" or "CD"

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