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
print(loanDf)
print(type(loanDf))


x = np.array(loanDf)
y = np.array(loanDf_y)

y = np.zeros(loanDf_y.shape) #设置因变量y。

''' 标签转换为0/1。将字典转换为数字 '''
y[loanDf_y == 'Y'] = 1


print(type(x))
print('x=', x)
print('y=', y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# clfr = RandomForestClassifier(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)
clfr = RandomForestRegressor(n_estimators=10, max_depth=None,min_samples_split=2, random_state=0)

'''
参数说明：
    n_estimators：在利用最大投票数或平均值来预测之前，你想要建立子树的数量。 较多的子树可以让模型有更好的性能，但同时让你的代码变慢。 你应该选择尽可能高的值，只要你的处理器能够承受的住，因为这使你的预测更好更稳定。default=10.
    criterion：判断节点是否继续分裂采用的计算方法，两类：● entropy ●  gini。其中Gini impurity衡量的是从一个集合中随机选择一个元素，基于该集合中标签的概率分布为元素分配标签的错误率。对于任何一个标签下的元素，其被分类正确的条件概率可以理解为在选择元素时选中该标签的概率与在分类时选中该标签的概率。基于上述描述，Gini impurity的计算就非常简单了，即1减去所有分类正确的概率，得到的就是分类不正确的概率。若元素数量非常多，切所有元素单独属于一个分类时，Gini不纯度达到极小值0。。
    max_depth ： （决策）树的最大深度，如果max_leaf_nodes参数指定，则忽略。两类：● int：深度 ● None：树会生长到所有叶子都分到一个类，或者某节点所代表的样本数已小于min_samples_split 。
    min_samples_split ： 分裂所需的最小样本数。两类：●int：样本数  ●  2：默认值
    random_state : 随机器对象。RandomState实例，或者为None,可选（默认值为None）RandomState ：如果是 int，random_state是随机数生成器使用的种子; 如果是RandomState实例，random_state就是随机数生成器; 如果为None，则随机数生成器是np.random使用的RandomState实例。
'''
scoresR = cross_val_score(clfr, x_train, y_train.ravel())
print('scoresR=',scoresR.mean())
clfr.fit(x_train, y_train.ravel())
print('clfr.feature_importances_=',clfr.feature_importances_)
#print(clfr.predict([[0, 0, 0, 0]]))

#y_train.values.ravel() 将会报错：AttributeError: 'numpy.ndarray' object has no attribute 'values'

'''测试结果的打印'''
answer = clfr.predict(x_test)
print('answer=', answer)

print('x_test=', x_test)
print('y_test=', y_test)
print('np.mean( answer == y_test)=', np.mean( answer == y_test))



'''
#准确率与召回率
precision, recall, thresholds = precision_recall_curve(y_train, clfr.predict(x_train))
answer = clfr.predict_proba(x)[:,1]
print('answer=', answer)
@print(classification_report(y, answer)) #, target_names = ['Y', 'N']
'''



#print ("AUC - ROC : ", roc_auc_score(y,clfr.oob_prediction))
print ("AUC - ROC : ", mean_absolute_error(y_test, answer))


# 循环打印5棵树
from sklearn.tree import export_graphviz
import pydot
# 从这1000个决策树中，我心情好，就选第6个决策树吧。
tree = clfr.estimators_[5]

#将决策树输出到dot文件中
feature_list = list(loanDf.columns)# 特征名列表
export_graphviz(tree,
                out_file = 'tree.dot',
                feature_names = feature_list,
                rounded = True,
                precision = 1)

# 将dot文件转化为图结构
(graph, ) = pydot.graph_from_dot_file('tree.dot')

#将graph图输出为png图片文件
graph.write_png('tree.png')


# 循环打印每棵树
# for idx, estimator in enumerate(clfr.estimators_):
#     # 导出dot文件
#     export_graphviz(estimator,
#                     out_file='tree{}.dot'.format(idx),
#                     feature_names=iris.feature_names,
#                     class_names=iris.target_names,
#                     rounded=True,
#                     proportion=False,
#                     precision=2,
#                     filled=True)
#     # 转换为png文件
#     os.system('dot -Tpng tree{}.dot -o tree{}.png'.format(idx, idx))