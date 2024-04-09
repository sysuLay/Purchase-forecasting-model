import lightgbm
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier, record_evaluation
from lightgbm import log_evaluation, early_stopping
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import cross_val_score

# data loading
train = pd.read_csv('train.csv')
test = pd.read_csv('To predict.csv')

# dataframe列元素种类分析
# for key in train.keys():
#     print(train[key].value_counts())
#     print(train[key].value_counts().index)

# 筛选后的列元素种类分析
print(train['subscribe'].value_counts())
print(train['subscribe'].value_counts().index)
print(train['job'].value_counts())
print(train['job'].value_counts().index)
print('--' * 50)
print(train['marital'].value_counts())
print(train['marital'].value_counts().index)
print('--' * 50)
print(train['education'].value_counts())
print(train['education'].value_counts().index)
print('--' * 50)
print(train['month'].value_counts())
print(train['month'].value_counts().index)
print('--' * 50)
print(train['housing'].value_counts())
print('--' * 50)
print(train['loan'].value_counts())

train.head()
train.describe()

# 无效数据剔除
train.drop(['id'], axis=1, inplace=True)
test.drop(['id'], axis=1, inplace=True)
d = {'yes': True, 'no': False}
train['subscribe'].replace(d, inplace=True)
d2 = {'mon': 1, 'tue': 2, 'wed': 3, 'thu': 4, 'fri': 5}
train['day_of_week'].replace(d2, inplace=True)
test['day_of_week'].replace(d2, inplace=True)
d4 = {'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'apr': 4, 'nov': 11, 'oct': 10, 'mar': 3, 'sep': 9, 'dec': 12}
train['month'].replace(d4, inplace=True)
test['month'].replace(d4, inplace=True)

# 分别对数值型和类别型变量中的缺失值进行填充
'''(由于lightgbm可以自动处理缺失值，处理方式与xgboost一致，
即在建树过程中，会尝试把当前特征中有缺失值的所有样本分别分到左子树或右子树，
然后看两种不同的分法中的哪以种可以让损失函数减少地更多，模型会记录下这种分法。
但是模型并不会对缺失值进行填充)'''
cat_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'poutcome']
num_cols = ['age', 'duration', 'campaign', 'pdays', 'previous', 'emp_var_rate', 'cons_price_index',
            'day_of_week', 'cons_conf_index', 'lending_rate3m', 'nr_employed', 'month']
sel_cat_cols = ['job', 'marital', 'loan']
sel_num_cols = ['duration', 'campaign', 'pdays', 'cons_price_index', 'cons_conf_index',
                'lending_rate3m', 'nr_employed']
sel1 = sel_num_cols + ['age', 'previous', 'emp_var_rate']
sel2 = ['duration', 'pdays', 'emp_var_rate', 'cons_price_index', 'campaign',
        'day_of_week', 'cons_conf_index', 'lending_rate3m', 'nr_employed', 'month']

# 相关性热力图
# plt.subplots(figsize=(20, 12))
# a1 = sns.heatmap(train[sel1].corr(), cmap="seismic", annot=True, vmin=-1, vmax=1, fmt='.1f', square=True)
# plt.show()

for col in sel_num_cols:
    d4 = {'unknown': train[col].mean()}
    train[col].replace(d4, inplace=True)
    test[col].replace(d4, inplace=True)

for col in sel_cat_cols:
    d3 = {'unknown': train[col].mode()[0]}
    train[col].replace(d3, inplace=True)
    test[col].replace(d3, inplace=True)

print("检验:")
print(train.keys())
print(len(train.keys()))

# 数据标准化
y1 = train['subscribe']
X1 = train[sel_num_cols]
X2 = test[sel_num_cols]
scaler_ss = StandardScaler()
X1 = scaler_ss.fit_transform(X1)

# 保存标化器
pickle.dump(scaler_ss, open('scaler.pkl', 'wb'))

# scaler your data,标化数据
scaler = pickle.load(open('scaler.pkl', 'rb'))
X2 = scaler.transform(X2)

train[sel_num_cols] = scaler_ss.fit_transform(X1)
test[sel_num_cols] = scaler_ss.fit_transform(X2)

print(X1[:5])
print(train[sel_num_cols].head())

# pca维度选取
X4 = train[sel2]
pca = PCA(n_components=0.9)
pca.fit(X4, y1)
ratio = pca.explained_variance_ratio_
print("pca.components_", pca.components_.shape)
print("pca_var_ratio", pca.explained_variance_ratio_.shape)

# # 绘制图形
# plt.plot([i for i in range(X4.shape[1])],
#          [np.sum(ratio[:i + 1]) for i in range(X4.shape[1])])
# plt.xticks(np.arange(X4.shape[1], step=5))
# plt.yticks(np.arange(0, 1.01, 0.05))
# plt.grid()
# plt.show()

# 特征升维
# train = train.drop(['default'], axis=1)
# train = train.drop(['housing'], axis=1)
# train =train.drop(['job'], axis=1)
# train =train.drop(['age'], axis=1)
train = pd.get_dummies(train, prefix_sep='_')
test = pd.get_dummies(test, prefix_sep='_')

print(train.keys())
print(len(train.keys()))
print(len(test.keys()))

# Modeling
y = train['subscribe']
X = train.drop(['subscribe'], axis=1)
# X = X.drop(['duration'], axis=1)

pca1 = PCA(n_components=6)
# pca1 = KernelPCA(n_components=5, kernel='rbf', gamma=15)
pca1.fit(X[sel2])
new_feature = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']

X[new_feature] = pca1.transform(X[sel2])
print(X[new_feature].head())
test[new_feature] = pca1.transform(test[sel2])

X.drop(X[sel2], axis=1, inplace=True)
test.drop(test[sel2], axis=1, inplace=True)

for key in X.keys():
    if key not in test.keys():
        test[key] = 0

print("数据对比：")
print(X.keys())
print(len(X.keys()))

# X.to_csv('Asd.csv', index=False)


# training，两种模型
clf = LGBMClassifier(objective='binary',
                     learning_rate=0.01,
                     n_estimators=100,
                     num_iterations=800,
                     max_depth=3,
                     bagging_fraction=0.8,
                     )

xgb = XGBClassifier(learning_rate=0.01,
                    n_estimators=125,  # 使用多少个弱分类器 200不如125
                    objective='binary:logistic',
                    booster='gbtree',
                    # max_depth=3,
                    # gamma=0,
                    # min_child_weight=1,
                    # max_delta_step=0,
                    # subsample=1,
                    # colsample_bytree=1,
                    # reg_alpha=0,
                    # reg_lambda=1,
                    )

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.1, random_state=42)
kf = KFold(n_splits=10, shuffle=True, random_state=8)
eval_result = {}
callbacks = [log_evaluation(period=80), early_stopping(stopping_rounds=20), record_evaluation(eval_result)]

acc_mean = []
for train_index, test_index in kf.split(X, y):
    X_train = X.iloc[train_index]
    y_train = y[train_index]
    X_test = X.iloc[test_index]
    y_test = y[test_index]
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.1)
    clf.fit(X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_names=['train', 'val'],
            eval_metric='auc',
            callbacks=callbacks
            )
    test_pred = clf.predict(X_test)
    acc_mean.append(accuracy_score(y_test, test_pred))
    print('accuracy:', accuracy_score(y_test, test_pred))
    # # 训练迭代图
    # plt.title('train_loss')
    # for data_name, metric_res in eval_result.items():
    #     for metric_name, log_ in metric_res.items():
    #         plt.plot(log_, label=f'{data_name}-{metric_name}',
    #                  color='steelblue' if 'train' in data_name else 'darkred',
    #                  linestyle=None if 'train' in data_name else '-.',
    #                  alpha=0.7)
    #
    # plt.legend()
    # plt.show()
    # plt.figure(figsize=[30,50],dpi=100)
    # lightgbm.plot_importance(clf, max_num_features=30)
    # plt.title("Featurertances")
    # plt.savefig('output.png', bbox_inches='tight')
    # plt.show()

print('10折交叉验证模型的平均准确率：')
print(sum(acc_mean) / len(acc_mean))

lgbm_mse_cv_scores = -cross_val_score(clf, X, y, cv=5, scoring="neg_log_loss", n_jobs=-1)
lgbm_rmse_score = np.sqrt(lgbm_mse_cv_scores)
xgbr_mse_cv_scores = -cross_val_score(xgb, X, y, cv=5, scoring='neg_log_loss', n_jobs=-1)
xgbr_rmse_score = np.sqrt(xgbr_mse_cv_scores)
print('XGBoost Regressor CV logloss Score :', xgbr_rmse_score.mean() / 3)
print('LightGBM Regressor CV logloss Score :', lgbm_rmse_score.mean() / 3)

print(cross_val_score(xgb, X, y, cv=5, scoring='accuracy', n_jobs=-1))
print(cross_val_score(clf, X, y, cv=5, scoring='accuracy', n_jobs=-1))
eval_result = {}
callbacks = [log_evaluation(period=80), early_stopping(stopping_rounds=20), record_evaluation(eval_result)]

# # testing
# test_pred = clf.predict(X_test)
# print('accuracy:', accuracy_score(y_test, test_pred))


# # 训练迭代图
# plt.title('train_loss')
# for data_name, metric_res in eval_result.items():
#     for metric_name, log_ in metric_res.items():
#         plt.plot(log_, label = f'{data_name}-{metric_name}',
#                 color='steelblue' if 'train' in data_name else 'darkred',
#                 linestyle=None if 'train' in data_name else '-.',
#                 alpha=0.7)
#
# plt.legend()
# plt.show()

# submit
sample = pd.read_csv('sample_submission.csv')
sample['subscribe'] = clf.predict(test).astype(bool)
d1 = {True: 'yes', False: 'no'}
sample['subscribe'].replace(d1, inplace=True)
sample.to_csv('submission.csv', index=False)
