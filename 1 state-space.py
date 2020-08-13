import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pylab as plt
from statsmodels.tsa.stattools import adfuller
import statsmodels.tsa.api as smt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import arma_order_select_ic
from scipy import stats
import itertools
from scipy.special import boxcox, inv_boxcox
from sklearn.metrics import mean_squared_error


# Load data

IO = '作业三数据.xlsx'
df = pd.read_excel(io=IO)

df.ds = pd.to_datetime(df.ds)

df.index = df.ds
df.drop(['ds'], axis=1, inplace=True)

# 检查数据中是否有缺失值，以下两种方式均可
# Flase:对应特征的特征值中无缺失值
# True：有缺失值
print(df.isnull().any())
print(np.isnan(df).any())

# 查看缺失值记录
train_null = pd.isnull(df)
train_null = df[train_null == True]
print(train_null)

# # 缺失值处理，以下两种方式均可
# # # 删除包含缺失值的行
# # df.dropna(inplace=True)
# # 缺失值填充
# df.fillna('0')

# 检查是否包含无穷数据
# False:包含
# True:不包含
print(np.isfinite(df).all())
# False:不包含
# True:包含
print(np.isinf(df).all())

# 数据处理
train_inf = np.isinf(df)
df[train_inf] = 0

# # 替换nan
# df[np.isnan(df)] = 0
# df = df.astype('int')

df.info()

# Resampling
df_month = df.resample('M').mean()
print(df_month.head())


# 拆分预测集及验证集
df_month_test = df_month[-170:]
print(df_month_test.tail())
print('df_month_test', len(df_month_test))
df_month = df_month[:-170]
print('df_month', len(df_month))

# PLOTS
fig = plt.figure(figsize=[15, 7])
plt.suptitle('sales, mean', fontsize=22)

plt.plot(df_month.y, '-', label='true-values_By Months')
plt.plot(df_month.original, '-', label='rew-data_By Months')
plt.legend()

# plt.tight_layout()
plt.show()


# 看趋势
plt.figure(figsize=[15, 7])
sm.tsa.seasonal_decompose(df_month.y).plot()
print("work3 test: p={}".format( adfuller(df_month.y)[1]))
# air_passengers test: p=0.996129346920727

# Box-Cox Transformations ts序列转换
df_month['y_box'], lmbda = stats.boxcox(df_month.y)
print("work3 test: p={}".format(adfuller(df_month.y_box)[1]))
# air_passengers test: p=0.7011194980409873

# Seasonal differentiation
# 季节性差分确定sax中m参数
df_month['y_box_diff'] = df_month['y_box'] - df_month['y_box'].shift(12)

# Seasonal differentiation
# 季节性差分确定sax中m参数
df_month['y_box_diff'] = df_month['y_box'] - df_month['y_box'].shift(12)

# Seasonal differentiation
# 季节性差分确定sax中m参数
df_month['y_box_diff'] = df_month['y_box'] - df_month['y_box'].shift(12)

# Initial approximation of parameters
Qs = range(0, 3)
qs = range(0, 3)
Ps = range(0, 3)
ps = range(0, 3)
D = 1
d = 1

parameters = itertools.product(ps, qs, Ps, Qs)
parameters_list = list(parameters)
# list参数列表
print('parameters_list:{}'.format(parameters_list))
print(len(parameters_list))

results = []
best_aic = float("inf")

for parameters in parameters_list:
    try:
        # SARIMAX 训练的时候用到转换之后的ts
        model = sm.tsa.statespace.SARIMAX(df_month.y_box, order=(parameters[0], d, parameters[1]),
                                          seasonal_order=(parameters[2], D, parameters[3], 12)).fit(disp=-1)
    except ValueError:
        print('wrong parameters:', parameters)
        continue

    aic = model.aic
    if aic < best_aic:
        best_model = model
        best_aic = aic
        best_param = parameters
    results.append([parameters, model.aic])

result_table = pd.DataFrame(results)
result_table.columns = ['parameters', 'aic']
print(result_table.sort_values(by='aic', ascending=True).head())
print(best_model.summary())
# Model:             SARIMAX(0, 1, 1)x(1, 1, 2, 12)


sm.graphics.tsa.plot_acf(best_model.resid[13:].values.squeeze(), lags=48)

# 下图是对残差进行的检验。可以确认服从正太分布，且不存在滞后效应。
best_model.plot_diagnostics(lags=30, figsize=(16, 12))

# df_month2 = df_month_test[['y']]
# # best_model.predict()  设定开始结束时间
# # invboxcox函数用于还愿boxcox序列
# df_month2['forecast'] = inv_boxcox(best_model.forecast(steps=5), lmbda)
# plt.figure(figsize=(15, 7))
# df_month2.y.plot()
# df_month2.forecast.plot(color='r', ls='--', label='Predicted Sales')
# plt.show()
#
# print(df_month2.forecast)

#df['forecast'] = inv_boxcox(best_model.predict(steps=5), lmbda)
#df['forecast'] = best_model.predict(steps=5)
# df_month2 = df_month_test[['y']]
# df_month2['forecast'] = inv_boxcox(best_model.forecast(steps=5), lmbda)
# print(df_month2.forecast)

df_month2 = df_month_test[['y']]
df_month2['forecast'] = inv_boxcox(best_model.forecast(steps=170), lmbda)
df_month2.fillna('0')
print(df_month2.forecast)


# 获取mse
print('mean_squared_error: {}'.format(mean_squared_error(df_month2.y, df_month2.forecast)/100))

# plot baseline and predictions
plt.figure(figsize=(20, 6))
l1, = plt.plot(df.original, color='red', linewidth=5, linestyle='--')
l2, = plt.plot(df.y, color='k', linewidth=4.5)
l3, = plt.plot(df_month2.forecast, color='g', linewidth=4.5)
plt.ylabel('Height m')
plt.legend([l1, l2, l3], ('raw-data', 'true-values', 'pre-values'), loc='best')
plt.title('state-space Prediction')
plt.show()
plt.savefig("state-space.png")