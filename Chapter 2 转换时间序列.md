# **Chapter 2 转换时间序列**

2.1在统计分析一个或一组时间序列之前，通常转换数据是适当的，序列的初始图会提供有关使用哪种转换的线索。有三种时间序列转换的一般类别：分布，平稳诱导和分解，这些通常可以结合起来为一个适当的变量来分析。

## **分布转换**

2.2许多统计程序对那些正态分布的数据更有效，或者至少是对称的且不过度峰度（胖尾），并且均值和方差近似恒定。观测到的时间序列通常需要某种形式的转换才能它们表现出这些分布特性，因为它们以“原始”形式存在通常不对称。例如，如果序列只能取正值（或至少非负值），则其分布通常会偏向右，因为尽管数据有一个自然的下限，通常为零，但没有上限，值可以“伸展”，可能无穷。在这种情况下，一种简单而流行的转换是取对数，通常以e为底（自然对数）。

```python
#数据来源：国家统计局
#http://data.stats.gov.cn/index.htm
df = pd.read_csv("国内生产总值.csv")
df['time'] = pd.to_datetime(df['time'],format='%Y')
df.set_index('time', inplace=True)

plt.plot(df['gdp'])
plt.show()
# 设置直方图的边框色
plt.hist(df['gdp'],edgecolor = 'black')
plt.show()
plt.hist(np.log(df['gdp']),edgecolor = 'black')
plt.show()
```

![GDP](media/GDP.png)

![GDP直方图](media/GDP直方图.png)

![GDP对数直方图](media/GDP对数直方图.png)

图2.1 中国国内生产总值（GDP），1952年到2019年，直方图及其对数的直方图。数据来自国家统计局。



2.3 图2.1显示中国1952年到2019年的国内生产总值（GDP）序列的水平，直方图及其对数的直方图。取对数明显降低了水平中的极右偏度，但它不会引起正态性，因为对数的分布是双峰的。其原因在图2.2中清楚可见，其中显示了GDP对数时间序列的图。分布的中心部分相对频率较低，在1980年代迅速过渡，因为中国实行改革开放，其特征是在此期间的序列斜率陡峭。显然，转换为对数不会引起平稳性，但是比较图2.2和图2.1的水平图，取对数确实“拉直”了趋势。取对数还可以稳定方差。图2.3绘制累积标准差比率$s_i(GDP)/s_i(logGDP)$定义为

$s_i^2=i^{-1}\sum_{t=1}^i(x_t-\bar{x}_i)^2,\bar{x}_i=i^{-1}\sum_{t=1}^ix_t$

由于该比率在整个观测期间中单调增加，对数变换显然有助于稳定方差，实际上，只要序列的标准偏差与其水平成比例，就可以做对数变换。

```python
plt.plot(np.log(df['gdp']))
plt.show()
```

![GDP对数](media/GDP对数.png)

图2.2 GDP的对数，1952年到2019年

```python
#计算累积标准差比率
a = [None,None]
for i in range(2,len(df['gdp']+1)):
    r = np.std(df['gdp'][0:i])
    a.append(r)
df['rate'] = np.array(a)
plt.plot(df['rate'])
plt.show()
```

![GDP比率](media/GDP比率.png)

图2.3累积标准偏差比率$s_i(GDP)/s_i(logGDP)$



2.4很明显，为了获得近似正态性，更一般的转换类别将很有用。一类包含对数作为特例的幂变换由Box和Cox（1964）提出，对于正数$x$：

$\begin{equation}f^{BC}(x_t,\lambda)=\left\{
\begin{aligned}
(x_t^\lambda-1)/\lambda & , & \lambda\neq0, \\
logx_t & , & \lambda=0.
\end{aligned}
\right.
\end{equation}$（2.1）



2.5 Box-Cox要求的正值限制可以通过多种方式放松转型。可以引入移位参数,以处理当x可能取负值，但仍有下界的情况，但当λ像2.6中估计时可能会导致推断方面的问题。可能的选择是Bickel和Doksum（1981）提出的符号幂变换：

$f^{SP}(x_t,\lambda)=(sgn(x_t)|x_t^\lambda|-1)/\lambda , \lambda>0$（2.2）

或Yeo和Johnson提出的广义幂（GP）转换：

$\begin{equation}f^{GP}(x_t,\lambda)=\left\{
\begin{aligned}
((x_t+1)^\lambda-1)/\lambda &  &x_t\ge0, \lambda\neq0 \\
log(x_t+1) &  & x_t\ge0,\lambda=0\\-((-x_t+1)^{2-\lambda}-1)/(2-\lambda) && x_t<0, \lambda\neq2\\-log(-x_t+1)&& x_t<0, \lambda=2
\end{aligned}
\right.
\end{equation}$（2.3）

另一种选择是Burbidge等人建议的反双曲正弦（IHS）变换，来处理任意符号的极端值：

$f^{IHS}(x_t,\lambda)=\frac{sinh^{-1}(\lambda x_t)}{\lambda}  =log \frac{\lambda x_t+(\lambda^2 x_t^2+1)^{1/2}}{\lambda} , \lambda>0$（2.4）



2.6变换参数λ可通过最大似然（ML）方法估计。假设对于一般变换$f(x_t,\lambda)$，假定模型$f(x_t,\lambda)=\mu_t+a_t$，其中$\mu_t$是$f(x_t,\lambda)$均值的一个模型，$a_t$是独立的并且服从零均值、恒定方差的正态分布。然后通过在λ上最大化下式集中对数似然函数获得ML估计量$\hat \lambda$：

$\ell(\lambda)=C_f-(T/2)\sum_{t=1}^Tlog\hat a_t^2+D_f(x_t,\lambda)$(2.5)

其中$\hat a_t^2=f(x_t,\lambda)-\hat \mu_t$是模型的ML估计的残差，$C_f$为常数，$D_f(x_t,\lambda)$取决于正在使用哪个变换：

$D_f(x_t,\lambda)=(\lambda-1)\sum_{t=1}^Tlog|x_t| \quad for(2.1),(2.2)\\ \qquad\qquad =(\lambda-1)\sum_{t=1}^Tsgn|x_t|log(|x_t|+1)\quad for(2.3)\\\qquad\qquad=-(1/2)\sum_{t=1}^Tlog(1+\lambda^2x_t^2)\quad for(2.4)$

如果$\hat \lambda$是ML估计量而$\ell(\hat \lambda)$是从(2.5)伴随的最大似然，则可以构造λ的置信区间，使用标准结果，$2(\ell(\hat \lambda)-\ell(\lambda))$渐近分布为$\chi^2(1)$，所以95％的置信区间，由符合$\ell(\hat \lambda)-\ell(\lambda)<1.92$的λ值确定。



## 平稳诱导变换

2.9一个简单的平稳性变换是采用连续的序列差，通过定义$x_t$的一阶差分为$\bigtriangledown x_t=x_t-x_{t-1}$。

在某些情况下，一阶差分可能不足以得到平稳，可能需要进一步的差分。



2.10进行高阶差分时需要谨慎。二阶差分定义为一阶差分的差分，即$\bigtriangledown \bigtriangledown x_t=\bigtriangledown ^2 x_t$。为提供明确的二阶差分表达式，方便起见引入滞后算子$B$，定义$B^jx_t\equiv x_{t-j}$，这样：

$\bigtriangledown x_t=x_t-x_{t-1}=x_t-Bx_t=(1-B)x_t$

因此：

$\bigtriangledown ^2x_t=(1-B)^2x_t=x_t-2x_{t-1}+x_{t-2}$

这显然不同于$x_t-x_{t-2}=\bigtriangledown _2x_t$，两周期差分，引入符号$=\bigtriangledown _j=1-B^j$，j周期差分。



2.11在某些时间序列中，通过采用部分或百分比变化，而不是简单的差分，也就是说，通过$\bigtriangledown x_t/x_{t-1}$或者$100\bigtriangledown x_t/x_{t-1}$变换。对于金融时间序列，这些是通常称为收益。



2.14变量的变化率与其对数之间存在有用的关系值得牢记，即：

$\frac{x_t-x_{t-1}}{x_{t-1}}=\frac{x_t}{x_{t-1}}-1\approx log\frac{x_t}{x_{t-1}}=logx_t-logx{t-1}=\bigtriangledown logx_t$ 

对于很小的y，从$log(1+y)\approx y$的事实得出近似值。因此，如果$y_t=(x_t-x_{t-1})/x_{t-1}$很小，通货膨胀率可以由对数的变化近似。



## **时间序列分解和平滑变换**

2.15通常情况下，将关注和注意力特别集中在时间序列的长期行为，然后，从短期行为中隔离这些“永久性”运动，更多“暂时性”波动，即通过分解分离观测，通常是“数据=拟合+残差”的形式。因为这样的分解更可能导出平滑的序列，这可能更好地认为“数据=平滑+粗糙”的形式，这是从Tukey（1977）借用的术语。Tukey本人喜欢滑动中位数，但滑动平均值（MA）成为迄今为止最流行的使时间序列平滑的方法。



2.16最简单的滑动平均将$x_t$替换为其自身，前值及其后值的平均值，即$MA(3)=(1/3)(x_{t-1}+x_t+x_{t+1})$。更复杂的公式:（2n+1)项加权和中心的MA[WMA（2n+1]将$x_t$替换为

$MA_t(2n+1)=\sum_{i=-n}^n\omega_ix_{t-i}$

其中权重$\omega_i$被限制和为1：$\sum_{i=-n}^n\omega_i=1$。权重经常与中心加权相似，因为权重数为奇数，所以$MA_t(2n+1)$与$x_t$相匹配，因此使用术语“中心的”。

随着MA中包含更多项，尽管它变得更加平滑，权衡在样本的开始和末尾，n个观测值都“丢失”了，n越大，将丢失更多的观测值。如果样本末处的观测值，即最近的，比开始时的样本更重要，那么可以考虑使用非中心的MA，如$\sum_{i=0}^n\omega_ix_{t-i}$，其中只有现在和过去的观测出现在MA中，因此，MA被称为“向后看”或“单边的”。



2.20 MA可以被解释为趋势；时间序列的长期，平滑演变的成分，即两成分分解的“平滑”部分。当一个观测序列的频率大于每年，例如每月或每季度，通常需要进行三成分分解。现在定义$X_t$，被分解为趋势项，$T_t$，季节项，$S_t$，和不规则项，$I_t$。分解可以做加法：

$X_t=T_t+S_t+I_t$

或乘法

$X_t=T_t\times S_t\times I_t$

尽管这种区别在某种程度上是人为的。季节性成分是一个规律的，短期的，每年的周期，而不规则成分是趋势和季节性成分确定后剩下的部分;因此，它应该是随机的，因此是不可预测的。

将经过季节性调整的序列定义为：

$X_t^{SA,A}=X_t-S_t=T_t+I_t$(2.13)

或

$X_t^{SA,M}=X_t/S_t=T_t\times I_t$X_t^{SA,A}=X_t-S_t=T_t+I_t$(2.14)$

取决于使用哪种分解形式。



2.21图1.4所示的啤酒销售序列的主要特征是突出的季节性销售模式。一种简单的季节性调整方法首先要使用MA估计趋势分量。由于啤酒的销量是每季度观察一次，可以用中心MA

$WMA_t(5)=(1/8)(x_{t-2}+x_{t+2})+(1/4)(x_{t-1}+x_t+x_{t+1})$

并假设加法分解[（2.11）](https://translate.googleusercontent.com/translate_f)，则可以获得“无趋势”序列，通过从观测序列中减去此MA：

这个无趋势的序列是季节性和非常规成分的总和，需要以某种方式解开它们。要做到这一点，“识别”假定$I_t$平均应为零（如果不是那么$I_t$的一部分将是可预测的，并且应该是趋势或季节成分的一部分）。这使得$S_t$可以通过取跨年度的每个季度的平均值来计算；例如，一季度季节因子为：

$S_t(Q1)=\frac{X_{1997Q1}+X_{1998Q1}+...+X_{2017Q1}}{16}$

其他三个季度的因子以类似的方式给出。这些因子的总和必须为零，因此，如果原始计算导致一个非零的和，就需要进行调整:如果这个和是$a\neq0$，那么应该从每个因子中减去$a/4$。

对于啤酒销量，这些因素经计算为：

$S_t(Q1)=-232.8, S_t(Q2)=184.1, S_t(Q3)=298.0, S_t(Q4)=-249.1$

趋势和季节成分如图2.14。啤酒销量的上升趋势清晰可见。这种方法使季节模式随时间推移是“恒定的”，可能在相对短的时期内足够，但更复杂的季节性调整程序可以季节性模式演变。

2.22 现在，通过“残差”计算出不规则项为：

$I_t=X_t-T_t-S_t$

并绘制如图2.14所示。它显然是随机的，并且因为它范围从-300到250万千升，相对较大，反映了许多因素会影响任何季度的啤酒销售。季节性调整啤酒销售量是根据(2.13)计算得出的，与未经调整的销售量一起显示在图2.15中。作为趋势和不规则的总和，经季节性调整序列反映了潜在趋势和随机性，有时甚至对啤酒销售产生冲击。

```python
#观测间隔为每季度
a = []
for i in range(2000,2016):
    for j in [1,4,7,10]:
        date = str(str(i)+'/'+str(j))
        a.append(date )

#数据来源：国家统计局
#http://data.stats.gov.cn/index.htm
df = pd.read_csv("啤酒销量.csv")

df['time'] = np.array(a)
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)
#通过啤酒销售量累计值(万千升)，计算每季度销量
b = []
for i in range(len(df['sale'])):
    if i%4 == 0:
        b.append( df['sale'][i])
    if i%4 != 0:
        b.append( df['sale'][i] - df['sale'][i-1])
df['sale'] = b
```

```python
#趋势分量
x = df['sale']
trend = [x[0],x[1]]
for i in range(2,len(x)-2):
    wma5 = (1/8)*(x[i-2]+x[i+2])+(1/4)*(x[i-1]+x[i]+x[i+1])
    trend.append(wma5)
trend.append(x[len(x)-2])
trend.append(x[len(x)-1])
df['trend'] = np.array(trend)
plt.plot(df['trend'] )
plt.show()
```

![趋势](media/趋势.png)

```python
#季节分量
a1 = []
a2 = []
a3 = []
a4 = []
for i in range(len(x)):
    if i%4 == 0:
        a1.append(df['sale'][i])
    elif i%4 == 1:
        a2.append(df['sale'][i])
    elif i%4 == 2:
        a3.append(df['sale'][i])
    elif i%4 == 3:
        a4.append(df['sale'][i])
s = np.mean(df['sale'])
season = [np.mean(a1)-s,np.mean(a2)-s,np.mean(a3)-s,np.mean(a4)-s]*int(len(x)/4)
#print(season)
df['season'] = np.array(season)
plt.plot(df['season'] )
plt.show()
```

![季节](media/季节.png)

```python
#不规则分量
df['irregular'] = df['sale'] - df['trend'] - df['season']
plt.plot(df['irregular'] )
plt.show()
```

![不规则](media/不规则.png)

图2.14 啤酒季度销售量的加法分解。三张图分别为趋势分量;季节分量；不规则分量。



```python
#经季节性调整的销量
df['nos'] = df['sale'] - df['season']
plt.plot(df['nos'],label="seasonally adjusted")
plt.plot(df['sale'],label="Observed")
plt.legend()
plt.show()
```

![调整](media/调整.png)

图2.15 观测和经季节性调整的中国啤酒销售量(万千升)。每季度，2000年至2015年。数据来自国家统计局。


