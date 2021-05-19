#!/usr/bin/env python
# coding: utf-8
# from  astropy.table import Table
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_boston
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor

# # Load data and data preprocessing
seed = 42  # for the same data division

kf = KFold(n_splits=5, random_state=seed, shuffle=True)
df_train = pd.read_csv('C:/Users/admin/Downloads/ML_chapter7_dataset/final/train_final.csv')
df_test = pd.read_csv('C:/Users/admin/Downloads/ML_chapter7_dataset/final/test_final.csv')

featureFlag = 6
print('featureFlag {:2d}'.format(featureFlag))


print('df_train',df_train.head(5))


if (featureFlag==0):  #//baseline
    X_train = df_train.drop(columns=['loan_status']).values
    Y_train = df_train['loan_status'].values.astype(int)
    X_test = df_test.drop(columns=['loan_status']).values
    Y_test = df_test['loan_status'].values.astype(int)


#------特征处理1---------------------------------------------------------
# 删除部分列
## 1 continuous_funded_amnt 与 continuous_loan_amnt完全相同，二者删其一
## 2 discrete_policy_code_1_one_hot 和 discrete_pymnt_plan_1_one_hot 取值唯一，信息熵为零
## 3 删除所有严重缺失列（缺失比例大于等于0.9）

if (featureFlag==1):
    train1, test1 = df_train.copy(), df_test.copy()
    train1.drop(columns='loan_status', inplace=True)
    test1.drop(columns='loan_status', inplace=True)

    del train1['continuous_funded_amnt'];
    del test1['continuous_funded_amnt']
    del train1['discrete_policy_code_1_one_hot'];
    del test1['discrete_policy_code_1_one_hot']
    del train1['discrete_pymnt_plan_1_one_hot'];
    del test1['discrete_pymnt_plan_1_one_hot']


    X_train = train1.values
    Y_train = df_train['loan_status'].values.astype(int)
    X_test = test1.values
    Y_test = df_test['loan_status'].values.astype(int)





#------特征处理2---------------------------------------------------------
# ### 2 基于业务理解构造变量
# ### var_business = (funded_amnt * int_rate + installment) / annual_inc
# #### 解释：分子体现贷款人所需偿还的债务，分母体现贷款人的经济实力，二者的比值体现其还贷能力

if (featureFlag==2):
    train2, test2 = df_train.copy(), df_train.copy()
    train2['cap'] = (df_train['continuous_funded_amnt'] * df_train['continuous_int_rate'] + df_train['continuous_installment']) / df_train['continuous_annual_inc']
    test2['cap']  = (df_test['continuous_funded_amnt'] * df_test['continuous_int_rate'] + df_test['continuous_installment']) / df_test['continuous_annual_inc']

    X_train = train2.values
    Y_train = df_train['loan_status'].values.astype(int)
    X_test = test2.values
    Y_test = df_test['loan_status'].values.astype(int)

#------特征处理2-end------------------------------------------------------
#------特征处理3(1+2)---------------------------------------------------------


if (featureFlag==3):
    train1, test1 = df_train.copy(), df_test.copy()
    train1.drop(columns='loan_status', inplace=True)
    test1.drop(columns='loan_status', inplace=True)

    del train1['continuous_funded_amnt'];
    del test1['continuous_funded_amnt']
    del train1['discrete_policy_code_1_one_hot'];
    del test1['discrete_policy_code_1_one_hot']
    del train1['discrete_pymnt_plan_1_one_hot'];
    del test1['discrete_pymnt_plan_1_one_hot']


    train1['cap'] = (df_train['continuous_funded_amnt'] * df_train['continuous_int_rate'] + df_train['continuous_installment']) / df_train['continuous_annual_inc']
    test1['cap'] = (df_test['continuous_funded_amnt'] * df_test['continuous_int_rate'] + df_test['continuous_installment']) / df_test['continuous_annual_inc']

    X_train = train1.values
    Y_train = df_train['loan_status'].values.astype(int)
    X_test = test1.values
    Y_test = df_test['loan_status'].values.astype(int)

#------特征处理3(1+2)-end---------------------------------------------------

#------特征处理4---------------------------------------------------------
# ### 添加衍生变量 -- 年收入与每月还款额的比值

if (featureFlag==4):
    train3, test3 = df_train.copy(), df_train.copy()

    train3['ratio_inc_installment'] = round(train3['continuous_annual_inc'] / train3['continuous_installment'])
    test3['ratio_inc_installment'] = round(test3['continuous_annual_inc'] / test3['continuous_installment'])

    X_train = train3.values
    Y_train = df_train['loan_status'].values.astype(int)
    X_test = test3.values
    Y_test = df_test['loan_status'].values.astype(int)
#------特征处理4-end---------------------------------------------------
#------特征处理5（1-4）---------------------------------------------------------
if (featureFlag==5):
    train1, test1 = df_train.copy(), df_test.copy()
    train1.drop(columns='loan_status', inplace=True)
    test1.drop(columns='loan_status', inplace=True)

    del train1['continuous_funded_amnt'];
    del test1['continuous_funded_amnt']
    del train1['discrete_policy_code_1_one_hot'];
    del test1['discrete_policy_code_1_one_hot']
    del train1['discrete_pymnt_plan_1_one_hot'];
    del test1['discrete_pymnt_plan_1_one_hot']

    train1['cap'] = (df_train['continuous_funded_amnt'] * df_train['continuous_int_rate'] + df_train[
        'continuous_installment']) / df_train['continuous_annual_inc']
    test1['cap'] = (df_test['continuous_funded_amnt'] * df_test['continuous_int_rate'] + df_test[
        'continuous_installment']) / df_test['continuous_annual_inc']

    train1['ratio_inc_installment'] = round(train1['continuous_annual_inc'] / train1['continuous_installment'])
    test1['ratio_inc_installment'] = round(test1['continuous_annual_inc'] / test1['continuous_installment'])

    X_train = train1.values
    Y_train = df_train['loan_status'].values.astype(int)
    X_test  = test1.values
    Y_test  = df_test['loan_status'].values.astype(int)
#------特征处理5-end---------------------------------------------------



#------特征处理6------------------------------------------------------
# ### 添加衍生变量--"从没逾期"("never_delinq")
if (featureFlag==6):
    train1, test1 = df_train.copy(), df_test.copy()
    train1.drop(columns='loan_status', inplace=True)
    test1.drop(columns='loan_status', inplace=True)

    del train1['continuous_funded_amnt'];
    del test1['continuous_funded_amnt']
    del train1['discrete_policy_code_1_one_hot'];
    del test1['discrete_policy_code_1_one_hot']
    del train1['discrete_pymnt_plan_1_one_hot'];
    del test1['discrete_pymnt_plan_1_one_hot']

    train1['cap'] = (df_train['continuous_funded_amnt'] * df_train['continuous_int_rate'] + df_train[
        'continuous_installment']) / df_train['continuous_annual_inc']
    test1['cap'] = (df_test['continuous_funded_amnt'] * df_test['continuous_int_rate'] + df_test[
        'continuous_installment']) / df_test['continuous_annual_inc']

    train1['ratio_inc_installment'] = round(train1['continuous_annual_inc'] / train1['continuous_installment'])
    test1['ratio_inc_installment'] = round(test1['continuous_annual_inc'] / test1['continuous_installment'])


    train1['never_delinq'] = train1['continuous_mths_since_last_major_derog'].isna()
    train1['never_delinq'] = train1['never_delinq'].map(lambda x: 0 if x else 1)
    test1['never_delinq'] = test1['continuous_mths_since_last_major_derog'].isna()
    test1['never_delinq'] = test1['never_delinq'].map(lambda x: 0 if x else 1)

    X_train = train1.values
    Y_train = df_train['loan_status'].values.astype(int)
    X_test  = test1.values
    Y_test  = df_test['loan_status'].values.astype(int)



# deleted_cols = []
# for col in train_data.columns:
#     if len(train_data[train_data[col].isna()]) / len(train_data) >= 0.9:
#         print(col)
#         deleted_cols.append(col)
# train1.drop(columns = deleted_cols, inplace=True)
# test1.drop(columns = deleted_cols, inplace=True)
#
# disc_cols, conti_cols = [], []
# for col in train1.columns:
#     if 'discrete' in col:
#         disc_cols.append(col)
#     elif 'continuous' in col:
#         conti_cols.append(col)
#
# disc_fea = ['addr_state', 'application_type', 'emp_length', 'grade','home_ownership', 'purpose', 'sub_grade', 'term']
#
# # #### a 单变量，离散情形，用Target Encoding
# from category_encoders.target_encoder import TargetEncoder
# encoder = TargetEncoder(cols = disc_fea,handle_unknown = 'value',handle_missing = 'value').fit(train1, train_data['loan_status'])
# TE_train = encoder.transform(train1)
# TE_test = encoder.transform(test1)
# TE_train['loan_status'] = train_data['loan_status']
# TE_test['loan_status'] = test_data['loan_status']
#------特征处理5-end-----------------------------------------------------



X_train.shape, Y_train.shape




# split data for five fold

five_fold_data = []

for train_index, eval_index in kf.split(X_train):
    x_train, x_eval = X_train[train_index], X_train[eval_index]
    y_train, y_eval = Y_train[train_index], Y_train[eval_index]

    five_fold_data.append([(x_train, y_train), (x_eval, y_eval)])


# # Algorithm
def get_model(param):
    model_list = []
    for idx, [(x_train, y_train), (x_eval, y_eval)] in enumerate(five_fold_data):
        print('{}-th model is training:'.format(idx))
        train_data = lgb.Dataset(x_train, label=y_train)
        validation_data = lgb.Dataset(x_eval, label=y_eval)
        bst = lgb.train(param, train_data, valid_sets=[validation_data])
        model_list.append(bst)
    return model_list


def get_model1(param):
    # 拟合构造CART回归树
    dtr = DecisionTreeRegressor()

    model_list = []
    for idx, [(x_train, y_train), (x_eval, y_eval)] in enumerate(five_fold_data):
        print('{}-th model is training:'.format(idx))

        x_train=removeNoneValue(x_train);
        y_train=removeNoneValue(y_train);

        bst = dtr.fit(x_train,  y_train)
        model_list.append(bst)
    return model_list

def get_model2(param):
    ## 决策树
    clf1 = DecisionTreeClassifier(max_depth=None, min_samples_split=2,random_state=0)
    # scores1 = cross_val_score(clf1, X, y)
    # print(scores1.mean())

    model_list = []
    for idx, [(x_train, y_train), (x_eval, y_eval)] in enumerate(five_fold_data):
        print('{}-th model is training:'.format(idx))
        x_train = removeNoneValue(x_train);
        y_train = removeNoneValue(y_train);
        bst = clf1.fit(x_train,  y_train)
        model_list.append(bst)
    return model_list


def get_model3(param):
    # ## 随机森林
    clf2 = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
    # scores2 = cross_val_score(clf2, X, y)
    # print(scores2.mean())

    model_list = []
    for idx, [(x_train, y_train), (x_eval, y_eval)] in enumerate(five_fold_data):
        print('{}-th model is training:'.format(idx))
        x_train = removeNoneValue(x_train);
        y_train = removeNoneValue(y_train);
        bst = clf2.fit(x_train,  y_train)
        model_list.append(bst)
    return model_list


def get_model4(param):
    # ## 逻辑回归
    clf = LogisticRegression(random_state=20)
    # lr_clf.fit(X_train, Y_train)
    # lr_clf.score(X_test, Y_test)

    model_list = []
    for idx, [(x_train, y_train), (x_eval, y_eval)] in enumerate(five_fold_data):
        print('{}-th model is training:'.format(idx))
        x_train = removeNoneValue(x_train);
        y_train = removeNoneValue(y_train);
        bst = clf.fit(x_train,  y_train)
        model_list.append(bst)
    return model_list



def get_model5(param):
    # ## 逻辑回归
    clf  = SVC()
    # clf.fit(X_train,Y_train)
    # clf.score(X_test,Y_test)

    model_list = []
    for idx, [(x_train, y_train), (x_eval, y_eval)] in enumerate(five_fold_data):
        print('{}-th model is training:'.format(idx))
        x_train = removeNoneValue(x_train);
        y_train = removeNoneValue(y_train);
        bst = clf.fit(x_train,  y_train)
        model_list.append(bst)
    return model_list

def get_model6(param):
    # ## 逻辑回归
    clf   = AdaBoostClassifier()
    # clf.fit(X_train,Y_train)
    # clf.score(X_test,Y_test)

    model_list = []
    for idx, [(x_train, y_train), (x_eval, y_eval)] in enumerate(five_fold_data):
        print('{}-th model is training:'.format(idx))
        x_train = removeNoneValue(x_train);
        y_train = removeNoneValue(y_train);
        bst = clf.fit(x_train,  y_train)
        model_list.append(bst)
    return model_list

def get_model7(param):
    # ###贝叶斯分类器
    clf = GaussianNB()
    # clf.fit(X_train, Y_train)
    # clf.score(X_test, Y_test)

    model_list = []
    for idx, [(x_train, y_train), (x_eval, y_eval)] in enumerate(five_fold_data):
        print('{}-th model is training:'.format(idx))
        x_train = removeNoneValue(x_train);
        y_train = removeNoneValue(y_train);
        bst = clf.fit(x_train,  y_train)
        model_list.append(bst)
    return model_list


def get_model8(param):
    # ###K近邻分类器
    clf = KNeighborsClassifier()
    # clf.fit(X_train, Y_train)
    # clf.score(X_test, Y_test)

    model_list = []
    for idx, [(x_train, y_train), (x_eval, y_eval)] in enumerate(five_fold_data):
        print('{}-th model is training:'.format(idx))
        x_train = removeNoneValue(x_train);
        y_train = removeNoneValue(y_train);
        bst = clf.fit(x_train,  y_train)
        model_list.append(bst)
    return model_list


# 将numpy.ndarray中的空值处理掉。 来不及写完该函数了。暂空，请助教老师可否抽空帮忙指导下@2021.05.19 19:40
def removeNoneValue(param):
    # 为解决错误：ValueError: Input contains NaN, infinity or a value too large for dtype('float32').
    print(type(param))
    print(param)
    param1 = np.array([filter(None, param)])
    print(type(param))
    print(param)

    return param;

# # test
def test_model(model_list):
    data = X_test
    five_fold_pred = np.zeros((5, len(X_test)))
    for i, bst in enumerate(model_list):
        ypred = bst.predict(data, num_iteration=bst.best_iteration)
        five_fold_pred[i] = ypred
    ypred_mean = (five_fold_pred.mean(axis=-2) > 0.5).astype(int)
    return accuracy_score(ypred_mean, Y_test)

def test_model1(model_list):
    data = X_test
    five_fold_pred = np.zeros((5, len(X_test)))
    for i, bst in enumerate(model_list):
        ypred = bst.predict(data, num_iteration=bst.best_iteration)
        five_fold_pred[i] = ypred
    ypred_mean = (five_fold_pred.mean(axis=-2) > 0.5).astype(int)
    return accuracy_score(ypred_mean, Y_test)





# # train

param_base = {'num_leaves': 31, 'objective': 'binary', 'metric': 'binary', 'num_round': 10}

param_fine_tuning = {'num_thread': 8, 'num_leaves': 128, 'metric': 'binary', 'objective': 'binary', 'num_round': 10,
                     'learning_rate': 3e-3, 'feature_fraction': 0.6, 'bagging_fraction': 0.8}

if (1==1): #baseline
    # base param train
    param_base_model = get_model(param_base)
    # param fine tuning
    param_fine_tuning_model = get_model(param_fine_tuning)

    base_score = test_model(param_base_model)
    fine_tuning_score = test_model(param_fine_tuning_model)
    print('base: {}, fine tuning: {}'.format(base_score, fine_tuning_score))
else:
    #------ -----------------------------------------------------
    # #  get_model1 -  8
    _model = get_model3(param_base)
    _fine_tuning_model = get_model3(param_fine_tuning)


    _score = test_model1(_model)
    _fine_tuning_score = test_model1(_fine_tuning_model)
    print('base: {}, fine tuning: {}'.format(_score, _fine_tuning_score))
    #------ -----------------------------------------------------





tabnet_clf = TabNetClassifier(
    optimizer_fn=torch.optim.Adam,
    optimizer_params=dict(lr=0.01),
    scheduler_params={"step_size":10,"gamma":0.9},
    scheduler_fn=torch.optim.lr_scheduler.StepLR,
    mask_type='sparsemax'
)
tabnet_clf.fit(X_train, Y_train, max_epochs=50, patience=5)
pre_test = tabnet_clf.predict(X_test)
accuracy_score(Y_test, pre_test)
