import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
def term(s):
    it = {' 60 months': 2, ' 36 months': 1}
    return it[s]

def application_type(s):
    it = {'Individual': 1, 'Joint App': 2 }
    return it[s]

def grade(s):
    it = {'A': 1, 'B': 2, 'C':3, 'D':4, 'E':5, 'F':6, 'G':7 }
    return it[s]

def sub_grade(s):
    it = {'A1': 11, 'B1': 21, 'C1':31, 'D1':41, 'E1':51, 'F1':61, 'G1':71,
          'A2': 12, 'B2': 22, 'C2': 32, 'D2': 42, 'E2': 52, 'F2': 62, 'G2': 72,
          'A3': 13, 'B3': 23, 'C3': 33, 'D3': 43, 'E3': 53, 'F3': 63, 'G3': 73,
          'A4': 14, 'B4': 24, 'C4': 34, 'D4': 44, 'E4': 54, 'F4': 64, 'G4': 74,
          'A5': 15, 'B5': 25, 'C5': 35, 'D5': 45, 'E5': 55, 'F5': 65, 'G5': 75
          }
    return it[s]

def loan_status(s):
    it = {'Default':0, 'Current': 1, 'Fully Paid': 2, 'Charged Off':3, 'In Grace Period':4, 'Late (16-30 days)':5, 'Late (31-120 days)':6}
    return it[s]

# 加载数据
sFile = 'C:/Users/admin/PycharmProjects/pythonProject/tools/LoanStats_2016Q2_tmp.csv'

loanDf = pd.read_csv(sFile, low_memory=False,
    usecols = ['loan_amnt',  'funded_amnt', 'term', 'installment', 'grade',
          'sub_grade',    'annual_inc',  'application_type', 'out_prncp','acc_now_delinq'
          ] ,# 'loan_status',
    converters={3:term,6:grade, 7:sub_grade,  50:application_type}
   )  #  'int_rate', 'emp_title','home_ownership','emp_length','issue_d',

loanDf_y = pd.read_csv(sFile,  usecols = ['loan_status'] ,converters={14:loan_status})

#增加衍生变量1:（贷款金额loan_amount/loan_amnt + 未偿还本金总额的余额out_prncp + 所欠月供installment + 借方拖欠债务的账户数目acc_now_delinq）/ 年收入annual_inc
loanDf['new_column'] = (loanDf['loan_amnt']+loanDf['out_prncp']+loanDf['installment']+loanDf['acc_now_delinq'])/loanDf['annual_inc']
x = loanDf
y = loanDf_y

x = np.array(loanDf)
y = np.array(loanDf_y)
# y = np.zeros(loanDf_y.shape) #设置因变量y。

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# 创建模型，训练模型
gbm = lgb.LGBMRegressor(objective='regression', num_leaves=31, learning_rate=0.05, n_estimators=20)

gbm.fit(X_train, y_train, eval_set=[(X_test, y_test.ravel() )], eval_metric='l1', early_stopping_rounds=5)


# 测试机预测
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration_)
# 模型评估
print('The rmse of prediction is:', mean_squared_error(y_test, y_pred) ** 0.5)
# feature importances
print('Feature importances:', list(gbm.feature_importances_))
print('-'*60)


# 网格搜索，参数优化
estimator = lgb.LGBMRegressor(num_leaves=31)
param_grid = {
    'learning_rate': [0.01, 0.1, 1],
    'n_estimators': [20, 40]
}
gbm = GridSearchCV(estimator, param_grid)
gbm.fit(X_train, y_train)
print('Best parameters found by grid search are:', gbm.best_params_)
print('-'*60)
