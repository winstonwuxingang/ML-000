# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.metrics import precision_recall_curve, roc_auc_score, mean_absolute_error
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import graphviz
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_recall_curve
from sklearn.preprocessing import StandardScaler
import warnings

# from AITrainCamp.LogisticRegression.DAPractice_LogisticRegression_CreditCard import show_metrics, plot_precision_recall,plot_confusion_matrix
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
    it = {'Default':0, 'Current': 1, 'Fully Paid': 2, 'Charged Off':3, 'In Grace Period':4, 'Late (16-30 days)':5, 'Late (31-120 days)':6, 'Issued':7,  }
    return it[s]

# sFile = 'C:/Users/admin/PycharmProjects/pythonProject/tools/LoanStats_2016Q2_tmp.csv'
sFile = './LendingCLub_all_3.csv'
print (sFile)

loanDf = pd.read_csv(sFile, low_memory=False,
    usecols = ['loan_amnt',  'funded_amnt', 'term', 'installment', 'grade','sub_grade',    'annual_inc'
          ] ,
    converters={3:term, 6:grade, 7:sub_grade}
   )  #  'int_rate', 'emp_title','home_ownership','emp_length','issue_d',


loanDf_y = pd.read_csv(sFile, usecols = ['debt_settlement_flag'])  # 'debt_settlement_flag', 'debt_settlement_flag_date', 'settlement_date', 'settlement_amount',
print('loanDf=',loanDf)
print(type(loanDf))
print('-'*80)
print('loanDf_y=',loanDf_y)
print(type(loanDf_y))
print('='*80)
print('loanDf_y.describe()=', loanDf_y.describe())
print('='*80)
print('loanDf_y.head()=',loanDf_y.head())
print(loanDf_y['debt_settlement_flag'].value_counts())


# describe
# sort(columns='x')
# a[a>0]=-a 表示将a中所有大于0的数转化为负值
# a['gender1']=a['gender'].astype('category')
# a['gender1'].cat.categories=['male','female']  #即将0，1先转化为category类型再进行编码。
# 描述性统计：
# 1.a.mean() 默认对每一列的数据求平均值；若加上参数a.mean(1)则对每一行求平均值；
#
# 2.统计某一列x中各个值出现的次数：a['x'].value_counts()；
#
# 3.对数据应用函数
# a.apply(lambda x:x.max()-x.min())
# 表示返回所有列中最大值-最小值的差。

print('-'*80)

x = np.array(loanDf)

y = np.zeros(loanDf_y.shape) #设置因变量y。
y = np.array(loanDf_y)

''' 标签转换为0/1。将字典转换为数字 '''
y[loanDf_y == 'Y'] = 1
y[loanDf_y == 'N'] = 0

def trans(x):
    if x == 'Y':
        return 1
    elif x == 'N':
        return 0
    else:
        return -1


loanDf_yy = loanDf_y['debt_settlement_flag'].apply(lambda x: trans(x))
loanDf_y['debt_settlement_flag2'] = loanDf_y['debt_settlement_flag'].apply(lambda x: trans(x))
print('='*80)
print('loanDf_yy.describe()=',loanDf_yy.describe())
print('loanDf_yy.head()=',loanDf_yy.head())
print('pd.Series(loanDf_yy)=', pd.Series(loanDf_yy))
print('='*80)
print('loanDf_y.describe()=',loanDf_y.describe())
print('loanDf_y.head()=',loanDf_y.head())
# print('pd.Series(loanDf_y)=', pd.Series(loanDf_y))
# print('loanDf_yy[\'debt_settlement_flag\'].value_counts()=', loanDf_yy[0].value_counts() ) #报错：AttributeError: 'numpy.int64' object has no attribute 'value_counts'


y = np.array(loanDf_yy)
y = np.array(loanDf_y['debt_settlement_flag2'])


print('x=', x)
print('-'*80)
print('y=', y)
print('-'*80)
print('type(y)=', type(y))
print('-'*80)
print( y > 0 )
print('np.max(y)=',np.max(y))
print('np.min(y)=',np.min(y))

# num = len(loanDf_y)
# num_fraud = len(loanDf_y[loanDf_y['debt_settlement_flag']=='Y'])
print(type(y))
num = len(y)
num_fraud = np.count_nonzero( y == 1 )
print('总记录笔数: ', num)
print('异常交易笔数：', num_fraud)
print('异常交易比例：{:.6f}'.format(num_fraud/num))


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

print('-'*80)
# 逻辑回归分类
clf = LogisticRegression()
clf.fit(x_train, y_train)
predict_y = clf.predict(x_test)
print('predict_y=', predict_y)
# 预测样本的置信分数
score_y = clf.decision_function(x_test)
print('score_y=', score_y)
print('-'*80)
# # 计算混淆矩阵，并显示
# cm = confusion_matrix(y_test, predict_y)
# class_names = [0,1]
# print('cm=', cm)
# print('class_names=', class_names)
# print('-'*80)
# # 显示混淆矩阵
# plot_confusion_matrix(cm, classes = class_names, title = '逻辑回归 混淆矩阵')
# # 显示模型评估分数
# print('-'*80)
# show_metrics()
# 计算精确率，召回率，阈值用于可视化
precision, recall, thresholds = precision_recall_curve(y_test, score_y)
print('-'*80)
print('precision=', precision)
print('recall=', recall)
print('thresholds=', thresholds)
# from AITrainCamp.LogisticRegression.DAPractice_LogisticRegression_CreditCard import plot_precision_recall, show_metrics
# plot_precision_recall()


# 计算混淆矩阵，并显示
cm = confusion_matrix(y_test, predict_y)
class_names = [0,1]
tp = cm[1,1]
fn = cm[1,0]
fp = cm[0,1]
tn = cm[0,0]
print('tp=',tp,'fn=',fn,'fp=',fp ,'tn=',tn)
print('精确率: {:.3f}'.format(tp/(tp+fp)))
print('召回率: {:.3f}'.format(tp/(tp+fn)))
print('F1值: {:.3f}'.format(2*(((tp/(tp+fp))*(tp/(tp+fn)))/((tp/(tp+fp))+(tp/(tp+fn))))))


# from AITrainCamp.LogisticRegression.DAPractice_LogisticRegression_CreditCard import plot_precision_recall, show_metrics