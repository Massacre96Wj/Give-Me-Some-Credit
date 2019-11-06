import pandas as pd
import numpy as np
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PowerTransformer
from sklearn.linear_model import LinearRegression,LassoCV,LogisticRegression
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.model_selection import KFold,train_test_split,StratifiedKFold,GridSearchCV,cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,accuracy_score, precision_score,recall_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

df0 = pd.read_csv('Give Me Some Credit/cs-training.csv')
df0 = df0.drop('Unnamed: 0',axis=1)
# 为方便查看调整列名为中文
df0.rename(columns = {'SeriousDlqin2yrs':'未来两年可能违约', 'RevolvingUtilizationOfUnsecuredLines':'可用信贷额度比例', 'age':'年龄',
       'NumberOfTime30-59DaysPastDueNotWorse':'逾期30-59天的笔数', 'DebtRatio':'负债率', 'MonthlyIncome':'月收入',
       'NumberOfOpenCreditLinesAndLoans':'信贷数量', 'NumberOfTimes90DaysLate':'逾期90天+的笔数',
       'NumberRealEstateLoansOrLines':'固定资产贷款数', 'NumberOfTime60-89DaysPastDueNotWorse':'逾期60-89天的笔数',
       'NumberOfDependents':'家属数量'},inplace=True)
# print(df0.info())
# print(df0.head().T)
# df0.describe().T

# 类别分布很不平衡，会影响建模效果
print(df0.未来两年可能违约.value_counts())

# 观察缺失值数,月收入 缺失29731，家属数量 缺失3924
df0.isnull().sum()

# 输出各字段分布情况图
# 大多数字段明显偏态，后续建模需考虑纠偏处理
plt.figure(figsize=(20,20),dpi=300)
plt.subplots_adjust(wspace =0.3, hspace =0.3)
for n,i in enumerate(df0.columns):
    plt.subplot(4,3,n+1)
    plt.title(i,fontsize=15)
    plt.grid(linestyle='--')
    df0[i].hist(color='grey',alpha=0.5)

# 通过箱型图观察各字段异常情况
# 负债率异常值（错误）较多；可用信贷额度比例 异常值（错误）较多，理论应小于或等于1
#  '逾期30-59天的笔数', '负债率', '月收入','逾期90天+的笔数', '固定资产贷款数', '逾期60-89天的笔数'异常值非常多，难以观察数据分布。
# 年龄方面异常值有待观察
plt.figure(figsize=(20,20),dpi=300)
plt.subplots_adjust(wspace =0.3, hspace =0.3)
for n,i in enumerate(df0.columns):
    plt.subplot(4,3,n+1)
    plt.title(i,fontsize=15)
    plt.grid(linestyle='--')
    df0[[i]].boxplot(sym='.')

# 由图可知，逾期笔数这三个字段，共线性极高，可考虑去除共线性
plt.figure(figsize=(10,5),dpi=300)
sns.heatmap(df0.corr(),cmap='Reds',annot=True)

#  构建异常值及明显错误处理函数
def error_processing(df):
    '''
    异常值处理，可根据建模效果，反复调节处理方案，建议谨慎删除数据。
    df：数据源
    '''

    def show_error(df, col, whis=1.5, show=False):
        '''
        显示上下限异常值数量，可选显示示例异常数据
        df：数据源
        col：字段名
        whis：默认1.5，对应1.5倍iqr
        show：是否显示示例异常数据
        '''
        iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
        upper_bound = df[col].quantile(0.75) + whis * iqr  # 上界
        lower_bound = df[col].quantile(0.25) - whis * iqr  # 下界
        # print(iqr,upper_bound,lower_bound)
        print('【', col, '】上界异常值总数：', df[col][df[col] > upper_bound].count())
        if show:
            print('异常值示例：\n', df[df[col] > upper_bound].head(5).T)
        print('【', col, '】下界异常值总数：', df[col][df[col] < lower_bound].count())
        if show:
            print('异常值示例：\n', df[df[col] < lower_bound].head(5).T)
        print('- - - - - - ')

    def drop_error(df, col):
        '''
        删除上下限异常值数量
        df：数据源
        col：字段名
        '''
        iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
        upper_bound = df[col].quantile(0.75) + 1.5 * iqr  # 上界
        lower_bound = df[col].quantile(0.25) - 1.5 * iqr  # 下界
        data_del = df[col][(df[col] > upper_bound) | (df[col] < lower_bound)].count()
        data = df[(df[col] <= upper_bound) & (df[col] >= lower_bound)]
        # print('总剔除数据量：',data_del)
        return data

    # 计数器
    n = len(df)

    # 可用信贷额度
    # 从分布直方图可知，比例大于1的应该为错误值。
    # 错误值共3321，若剔除可能影响建模效果。剔除>=20000的数据
    show_error(df, '可用信贷额度比例')
    df = df[df.可用信贷额度比例 <= 20000]

    # 年龄
    # 异常值数量不多，剔除年龄大于100小于18的异常数据
    show_error(df, '年龄')
    df = df[(df['年龄'] > 18) & (df['年龄'] < 100)]

    # 逾期30-59天的笔数
    # 根据箱型图去除>80的异常数据
    show_error(df, '逾期30-59天的笔数')
    df = df[df['逾期30-59天的笔数'] < 80]

    # 逾期90天+的笔数
    # 根据箱型图去除>80的异常数据
    show_error(df, '逾期90天+的笔数')
    df = df[df['逾期90天+的笔数'] < 80]

    # 逾期60-89天的笔数
    # 根据箱型图去除>80的异常数据
    show_error(df, '逾期60-89天的笔数')
    df = df[df['逾期60-89天的笔数'] < 80]

    # 负债率
    # 根据箱型图去除>100000的异常数据
    show_error(df, '负债率')
    df = df[df['负债率'] < 100000]

    # 月收入
    # 根据箱型图去除>500000的异常数据
    show_error(df, '月收入')
    df = df[(df['月收入'] < 500000) | df.月收入.isna()]

    # 固定资产贷款数
    # 根据箱型图去除>20的异常数据
    show_error(df, '固定资产贷款数')
    df = df[df['固定资产贷款数'] < 20]

    # 家属数量
    # 根据箱型图去除>10的异常数据
    show_error(df, '家属数量')
    df = df[(df['家属数量'] < 12) | df.家属数量.isna()]

    # 信贷数量 - 保留异常值

    print('共删除数据 ', n - len(df), ' 条。')

# 构建去共线性函数
# 3种违约情况，从上节的相关系数热力图中，可以看出有很高的共线性
# 可考虑保留'逾期90天+的笔数'，求出'逾期60-89天的笔数'/'逾期30-59天的笔数'的比值
def collineation_processing(df, col, col1, col2, name):
    '''
    去除共线性，保留一个字段，其他字段求比值
    df：数据源
    col：保留字段
    col1，col2：求比值字段
    name：新比值字段名称
    '''

    def trans2percent(row):
        if row[col2] == 0:
            return 0
        else:
            return row[col1] / row[col2]

    df[name] = df.apply(trans2percent, axis=1)
#     df[[name,col]].corr()

# collineation_processing(df,'逾期90天+的笔数'，'逾期60-89天的笔数'，'逾期30-59天的笔数'，'逾期60-89天/30-59天')

# 构建缺失值处理函数
def missing_values_processing(df, func1=1, func2=1):
    '''
    缺失值处理
    df：数据源
    func1：默认为1，众数填充家属；0，去除带空值数据行。
    func2：默认为1，众数填充月收入；0，平均数填充月收入。
    '''
    # 家属数量 - 剔除或众数填充
    if func1 == 1:
        df.loc[df.家属数量.isna(), '家属数量'] = df.家属数量.mode()[0]
    elif func1 == 0:
        df = df.dropna(subset=['家属数量'])
    else:
        print('parameter wrong!')

    # 月收入 - 剔除或均值填充
    if func1 == 1:
        df.loc[df.月收入.isna(), '月收入'] = df.月收入.mode()[0]
    elif func1 == 0:
        df.loc[df.月收入.isna(), '月收入'] = df.月收入.mean()[0]
    else:
        print('parameter wrong!')

    # 可考虑建模填充 月收入，构建回归模型性能查看函数（最终测试结果很不理想）
#     def perfomance_reg(model,X,y,name=None):
#         y_predict = model.predict(X)
#         check = pd.DataFrame(y)
#         check['y_predict'] = y_predict
#         check['abs_err'] = abs(check['y_predict'] - check[y.name] )
#         check['ape'] = check['abs_err'] / check[y.name]
#         ape = check['ape'][check['ape']!=np.inf].mean()
#         if name:
#             print(name,':')
#         print(f'mean squared error is: {mean_squared_error(y,y_predict)}')
#         print(f'mean absolute error is: {mean_absolute_error(y,y_predict)}')
#         print(f'R Squared is: {r2_score(y,y_predict)}')
#         print(f'mean absolute percent error is: {ape}')
#         print('- - - - - - ')

# 线性回归填充月收入,mae较大
#     train_x = df1[df1.月收入.notna()].drop(['逾期30-59天的笔数','逾期60-89天的笔数','月收入'],axis=1)
#     train_y = df1[df1.月收入.notna()].月收入
#     test_x = df1[df1.月收入.isna()].drop(['逾期30-59天的笔数','逾期60-89天的笔数','月收入'],axis=1)
#     pipe_lr = Pipeline([
#             ('sc',StandardScaler()),
#             ('pow_trans',PowerTransformer()),
#             ('rf',LinearRegression())
#             ])
#     pipe_lr.fit(train_x,train_y)
#     perfomance_reg(pipe_lr,train_x,train_y)
#     pipe_lr.predict(test_x)

# 随机森林填充月收入，表现较线性回归略好一点,但也很差
#     dd = df1[df1.月收入.notna()].sample(n=5000)
#     train_x_sample = dd.drop(['逾期30-59天的笔数','逾期60-89天的笔数','月收入'],axis=1)
#     train_y_sample = dd.月收入
#     test_x_sample = df1[df1.月收入.isna()].drop(['逾期30-59天的笔数','逾期60-89天的笔数','月收入'],axis=1)
#     pipe_rf = Pipeline([
#             ('sc',StandardScaler()),
#             ('pow_trans',PowerTransformer()),
#             ('rf',RandomForestRegressor(criterion='mae',n_estimators=200,verbose=1,n_jobs=-1))
#             ])
#     pipe_rf.fit(train_x_sample,train_y_sample)
#     perfomance_reg(pipe_rf,train_x_sample,train_y_sample)
#     df1.loc[df1.月收入.isna(),'月收入'] = pipe_rf.predict(test_x_sample)

# 构建重采样函数
# 从数据初探可以发现，'未来两年可能违约'标签类别分布不均，需对样本进行重取样
def resample(df):
    '''
    使样本'未来两年可能违约'标签的0，1项可以各占一半，以提高预测效果。sample()可以考虑添加random_state以便生成相同样本集
    df：数据源
    '''
    num = df['未来两年可能违约'].value_counts()[1]
    df_t = df[df.未来两年可能违约==1]
    df_f = df[df.未来两年可能违约==0].sample(frac=1)[0:num]
    df_balanced = pd.concat([df_t,df_f]).sample(frac=1).reset_index(drop=True)
#     print(df_balanced.未来两年可能违约.value_counts())
    return df_balanced


# 数据预处理&模型训练
# 数据预处理 ==> 数据划分 ==> 模型训练及参数搜索

# 设 【 df1 】 为违约概率模型建模所用数据集
df1 = df0.copy()

# 异常处理
error_processing(df1)
# 去除共线性
collineation_processing(df1,'逾期90天+的笔数', '逾期60-89天的笔数', '逾期30-59天的笔数','逾期60-89天/30-59天')
# 缺失值处理
missing_values_processing(df1,func1=1,func2=1)
# 数据重采样
df_balanced = resample(df1)

# 最后将数据集划分成训练集和验证集，两者划分比例都为8：2
# 可考虑删去的列：'逾期30-59天的笔数','逾期60-89天的笔数','逾期90天+的笔数','逾期60-89天/30-59天','未来两年可能违约'
X = df_balanced.drop(['未来两年可能违约','逾期60-89天/30-59天'],axis=1)
y = df_balanced['未来两年可能违约']
xtrain,xtest,ytrain,ytest = train_test_split(X,y,test_size=0.2)    # random_state=42

# 分层k折交叉拆分器 - 用于网格搜索
cv = StratifiedKFold(n_splits=3,shuffle=True)

# 分类模型性能查看函数
def perfomance_clf(model,X,y,name=None):
    y_predict = model.predict(X)
    if name:
        print(name,':')
    print(f'accuracy score is: {accuracy_score(y,y_predict)}')
    print(f'precision score is: {precision_score(y,y_predict)}')
    print(f'recall score is: {recall_score(y,y_predict)}')
    print(f'auc: {roc_auc_score(y,y_predict)}')
    print('- - - - - - ')

# 逻辑回归模型
# 参数设定
log_params = {"penalty":['l1','l2'],
                 'C':[0.001*10**i for i in range(0,7)]}
# 参数搜索
log_gridsearch = GridSearchCV(LogisticRegression(solver='liblinear'),log_params,cv=cv,
                               n_jobs=-1,scoring='roc_auc',verbose=2,refit=True)
# 工作流管道
pipe_log = Pipeline([
        ('sc',StandardScaler()),    # 标准化Z-score
        ('pow_trans',PowerTransformer()),    # 纠偏
        ('log_grid',log_gridsearch)
        ])
# 搜索参数并训练模型
pipe_log.fit(xtrain,ytrain)
# 最佳参数组合
print(pipe_log.named_steps['log_grid'].best_params_)
# 训练集性能指标
perfomance_clf(pipe_log,xtrain,ytrain,name='train')
# 测试集性能指标
perfomance_clf(pipe_log,xtest,ytest,name='test')
# 交叉验证查看平均分数
cross_val_score(pipe_log,xtrain,ytrain,cv=3,scoring='roc_auc').mean()

# 随机森林分类模型
# 随机森林分类模型
rf_clf = RandomForestClassifier(criterion='gini',
                               n_jobs=-1,
                               n_estimators=1000)    # random_state
# 参数设定
rf_grid_params = {'max_features':['auto'],    # ['auto',0.5,0.6,0.9] 未知最优参数时可以自己设定组合
                    'max_depth':[6,9]}    # [3,6,9]
# 参数搜索
rf_gridsearch = GridSearchCV(rf_clf,rf_grid_params,cv=cv,
                               n_jobs=-1,scoring='roc_auc',verbose=10,refit=True)
# 工作流管道
pipe_rf = Pipeline([
        ('sc',StandardScaler()),
        ('pow_trans',PowerTransformer()),
        ('rf_grid',rf_gridsearch)
        ])
# 搜索参数并训练模型
pipe_rf.fit(xtrain,ytrain)
# 最佳参数组合
print(pipe_rf.named_steps['rf_grid'].best_params_)
# 训练集性能指标
perfomance_clf(pipe_rf,xtrain,ytrain,name='train')
# 测试集性能指标
perfomance_clf(pipe_rf,xtest,ytest,name='test')
# 注意！！！交叉验证查看平均分数（由于管道会反复搜索参数，会较耗时）
cross_val_score(pipe_rf,xtrain,ytrain,cv=3,scoring='roc_auc').mean()

# xgboost模型
# xgboost模型
xgb_clf = xgb.XGBClassifier(objective='binary:logistic',
                            n_job=-1,
                            booster='gbtree',
                            n_estimators=1000,
                            learning_rate=0.01)
# 参数设定
xgb_params = {'max_depth':[6,9],    # 注意参数设置，数量多了会更加耗时
             'subsample':[0.6,0.9],
             'colsample_bytree':[0.5,0.6],
             'reg_alpha':[0.05,0.1]}
# 参数搜索
xgb_gridsearch = GridSearchCV(xgb_clf,xgb_params,cv=cv,n_jobs=-1,
                                 scoring='roc_auc',verbose=10,refit=True)
# 工作流管道
pipe_xgb = Pipeline([
    ('sc',StandardScaler()),
    ('pow_trans',PowerTransformer()),
    ('xgb_grid',xgb_gridsearch)
])
# 搜索参数并训练模型
pipe_xgb.fit(xtrain,ytrain)
# 最佳参数组合
print(pipe_xgb.named_steps['xgb_grid'].best_params_)
# 训练集性能指标
perfomance_clf(pipe_xgb,xtrain,ytrain,name='train')
# 测试集性能指标
perfomance_clf(pipe_xgb,xtest,ytest,name='test')
# 注意！！！交叉验证查看平均分数（由于管道会反复搜索参数，会很耗时）
cross_val_score(pipe_xgb,xtrain,ytrain,cv=3,scoring='roc_auc').mean()

# 查看字段相对xgboost模型的重要程度
# 一般显示 ['可用信贷额度比例', '年龄', '负债率', '月收入', '信贷数量'] 这些字段比较重要
plt.figure(figsize=(10,5))
ax = plt.subplot(1,1,1)
xgb.plot_importance(pipe_xgb.named_steps['xgb_grid'].best_estimator_,
                       max_num_features=40,height=0.5,grid=False,ax=ax)
xtrain.columns


# 预测并生成结果
# 预测集数据读取与处理
dftest = pd.read_csv('Give Me Some Credit/cs-test.csv').drop('Unnamed: 0',axis=1)
dftest.rename(columns = {'SeriousDlqin2yrs':'未来两年可能违约', 'RevolvingUtilizationOfUnsecuredLines':'可用信贷额度比例', 'age':'年龄',
      'NumberOfTime30-59DaysPastDueNotWorse':'逾期30-59天的笔数', 'DebtRatio':'负债率', 'MonthlyIncome':'月收入',
      'NumberOfOpenCreditLinesAndLoans':'信贷数量', 'NumberOfTimes90DaysLate':'逾期90天+的笔数',
      'NumberRealEstateLoansOrLines':'固定资产贷款数', 'NumberOfTime60-89DaysPastDueNotWorse':'逾期60-89天的笔数',
      'NumberOfDependents':'家属数量'},inplace=True)
dftest.loc[dftest.家属数量.isna(),'家属数量'] = df1.家属数量.mode()[0]
dftest.loc[dftest.月收入.isna(),'月收入'] = df1.月收入.mode()[0]

# 以xgboost模型预测，生成csv结果文件
result = pipe_xgb.predict_proba(dftest.drop('未来两年可能违约',axis=1))
result_ = [[n+1,i] for n,i in enumerate(result[:,1])]
df_result = pd.DataFrame(result_,columns=['Id','Probability'])
df_result.to_csv('sampleEntry.csv',index=False)

# 模型保存方法
import pickle
with open('pipe_log.pickle','wb') as f:
    pickle.dump(pipe_log,f)
with open('pipe_log.pickle','rb') as f:
    clf = pickle.load(f)

