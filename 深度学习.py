import numpy
import matplotlib.pyplot as plt
from pandas import read_excel
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# read data
data = read_excel('作业三数据.xlsx')
values1 = data['newseries'].values
dataset = values1.reshape(-1,1) # 将一维数组转化为二维数组
dataset = dataset.astype('float32') # 将数据转化为32位浮点型，防止0数据

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back = 1): # 后一个数据和前look_back个数据有关系
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i+look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)  # 生成输入数据和输出数据

# 随机数生成时算法所用开始的整数值
numpy.random.seed(7)

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1)) # 归一化0-1
dataset = scaler.fit_transform(dataset)

# split into train and test sets  #训练集和测试集分割
train_size = int(len(dataset) * 0.70) # 70%的训练集，剩下测试集
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]  # 训练集和测试集

# use this function to prepare the train and test datasets for modeling
look_back = 1
trainX, trainY = create_dataset(train, look_back) # 训练输入输出
testX, testY = create_dataset(test, look_back) # 测试输入输出

 #reshape input to be [samples, time steps, features]#注意转化数据维数
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# 建立LSTM模型
model = Sequential()
model.add(LSTM(11, input_shape=(1, look_back))) # 隐层11个神经元 （可以断调整此参数提高预测精度）
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')# 评价函数mse，优化器adam
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2) # 100次迭代
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# 数据反归一化
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])


trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))


trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict

# plot baseline and predictions
plt.figure(figsize=(20, 6))
l1, = plt.plot(scaler.inverse_transform(dataset), color='red', linewidth=5, linestyle='--')
l2, = plt.plot(trainPredictPlot, color='k', linewidth=4.5)
l3, = plt.plot(testPredictPlot, color='g', linewidth=4.5)
plt.ylabel('Height m')
plt.legend([l1, l2, l3], ('raw-data', 'true-values', 'pre-values'), loc='best')
plt.title('LSTM Gait Prediction')
plt.show()
plt.savefig("LSTM.png")