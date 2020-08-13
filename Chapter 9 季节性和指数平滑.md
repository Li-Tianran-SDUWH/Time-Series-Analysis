# **Chapter 9 季节性和指数平滑**

## **时间序列的季节性模式**

9.1 在2.16‑2.17中，我们引入了季节性模式的想法，以大于每年（通常每月或每季）的频率观测到的时间序列。季节性的存在通常会立即从序列图显示，但它也会在样本自相关函数（SACF）中体现出，用适当差分的数据。图9.1显示了一阶差分啤酒销量的SACF，被明显的季节模式主导。显然，季节性模式是这些序列的特征，因此很容易进行显式建模或通过适当的季节性调整程序删除。

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
#观测间隔为每季度
a = []
for i in range(2000,2016):
    for j in [1,4,7,10]:
        date = str(str(i)+'/'+str(j))
        a.append(date )

#数据来源：国家统计局
#http://data.stats.gov.cn/index.htm
df = pd.read_csv("E:\\bai\\Py_19\\啤酒销量.csv")

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

```
#一阶差分
sale_diff = df['sale'].diff().dropna()
#SACF
pacf = plot_acf(sale_diff, lags=25)
plt.title("SACF")
pacf.show()
```

![acf](media/acf.png)

图9.1 中国啤酒销售量(万千升)一阶差分的SACF，每季度，2000年至2015年。数据来自国家统计局。



## **对确定季节性建模**

9.2 在2.6-2.7中提到了一个简单的季节性模型。使用“季节性均值”模型，其中每个季节都有不同的均值，$x_t$的模型为：

$x_t=\sum_{i=1}^m\alpha_is_{i,t}+\varepsilon_t$(9.1)

其中季节性虚拟变量$s_{i,t}$在第i个季节取值1，否则为零，一年中有m个季节。因此，例如，如果数据是每月,则1月$i=1$，等等，以及m=12。噪声$\varepsilon_t$可以建模为ARIMA过程，$\phi(B)\varepsilon_t=\theta(B)a_t$，如果需要的话。因此，回归模型(9.1)假设季节性模式是确定的，在这种意义上说，季节性均值$\alpha_i,i=1,...,m$随时间保持恒定。



## **对随机季节性建模**

9.3 但是，排除季节性模式演变的可能性是不明智的：换句话说，存在随机季节性。与随机趋势建模一样，发现ARIMA过程在建模随机季节性方面做得很好。



9.4尝试用ARIMA模型建模季节性时间序列时的重要考虑因素是，确定哪种过程能最好地匹配表征数据的SACF和PACF。看啤酒销售序列，我们已经注意到$\triangledown x_t$的SACF中的季节性模式，如图9.1所示。在进一步考虑SACF时，我们注意到季节性表现，在季节性时滞$(4k,k\ge1)$有很大的正自相关性，在“卫星”$[4(k-1),4(k+1)]$有负自相关性。这些季节性自相关的缓慢下降说明季节非平稳性，类似于“非

季节非平稳”，这可以通过季节差分来解决，即通过将$\triangledown_4=1-B^4$运算符与常规运算符$\triangledown$结合使用。$\bigtriangledown _j=1-B^j$，为j周期差分。图9.3显示了啤酒销售$\triangledown\triangledown_4$转换的SACF，现在显然是平稳的，因此可能适合ARIMA识别。

```python
#四周期差分
sale_diff = df['sale'].diff(4).dropna()
#一阶差分
sale_diff = sale_diff.diff().dropna()
pacf = plot_acf(sale_diff, lags=25)
plt.title("SACF")
pacf.show()
```

![acf2](media/acf2.png)

图9.3  $\triangledown\triangledown_4$转换的啤酒销售量的SACF 



9.5通常，如果我们的季节周期为m，差分算子可以表示为$\triangledown_m$。非季节性和季节性差分运算符可以分别应用d和D次，因此季节性ARIMA模型可以采用一般形式

$\triangledown^d\triangledown_m^Dx_t=\theta(B)a_t$（9.2）

适当形式的$\theta(B)$和$\phi(B)$多项式，至少在原理上，可以通过通常的识别和/或模型选择获得。不幸的是，通常会遇到两个困难。首先，季节模型的PACF既难以推导又难以解释，因此常规识别通常仅基于合适的SACF的行为。其次，由于$\theta(B)$和$\phi(B)$多项式需要考虑到季节自相关，其中至少一个必须为最小阶m，这通常意味着在模型选择程序中需要考虑的模型数量可能会变得过大。



9.6 图9.3充分说明了这一困难。许多SACF中显示的值与零显著不同，因此确定(9.2)类型的模型实际上是不可能的，如果实现，无疑难以解释。Box和Jenkins（1970）

因此提出了使用(9.2)受限版本的论点，他们认为这可以为许多季节时间序列提供充足的拟合。

```python
#前十年啤酒销售的观测（2000‑2009），按季度排列
q1 = []
q2 = []
q3 = []
q4 = []
for i in range(40):
    if i%4 == 0:
        q1.append(df['sale'][i])
    elif i%4 == 1:
        q2.append(df['sale'][i])
    elif i%4 == 2:
        q3.append(df['sale'][i])
    elif i%4 == 3:
        q4.append(df['sale'][i])
df_1 = pd.DataFrame({
    'Q1':q1,'Q2':q2,'Q3':q3,'Q4':q4,
})
df_1['time'] = np.array([i for i in range(2000,2010)])
df_1.set_index('time', inplace=True)
print(df_1)
```

![观测](media/观测.png)

9.7通过介绍此模型，请考虑前十年啤酒销售的观测（2000‑2009），按季度排列，这强调了一个事实，在季节性数据中，有不是一个，而是两个，重要的时间间隔。

这些间隔在这里对应于季度和年份，我们因此，预期会发生两种关系：（1）每年连续的几个季度的观测之间，以及（2）连续几年相同的季度的相同观测值之间。这在数据中很明显，季节效应意味着对某个特定季度（例如第四季度）的观测与之前第四季度的观测相关。



9.8然后可以通过以下形式的模型将第四季度的观测连接起来：

$\Phi(B^m)\triangledown_m^Dx_t=\Theta(B^m)\alpha_t$(9.3)

在示例中m=4，$\Phi(B^m)$和$\Theta(B^m)$分别$B^m$是的P和Q阶多项式，即

$\Phi(B^m)=1-\Phi_1B^m-\Phi_2B^{2m}-...-\Phi_PB^{Pm}$

$\Theta(B^m)=1-\Theta_1B^m-\Theta_2B^{2m}-...-\Theta_QB^{Qm}$

满足标准平稳性和可逆性条件。现在假设相同的模型适用于每个季度的观察。这意味着与不同年份的固定季度相对应的所有误差均不相关。但是，相邻季度对应的误差不需要不相关，即，误差序列$\alpha_t,\alpha_{t-1},...$可能是自相关的。例如，2009年第四季度的啤酒销售量与之前的第四季度相关，也将与2009年第三季度，第二季度等相关。自相关可以通过第二个非季节性的过程来建模：

$\phi(B)\triangledown^d\alpha_t=\theta(B)a_t$(9.4)

使得$\alpha_t$是ARIMA（p，d，q），$a_t$是白噪声过程。将(9.4)代入(9.3)可得出一般的乘积季节性模型：

$\phi_p(B)\Phi_P(B^m)\triangledown^d\triangledown_m^Dx_t=\theta_q(B)\Theta_Q(B^m)a_t$(9.5)

为了清楚起见添加下标p，P，q，Q，以便可以强调各种多项式的阶，并且ARIMA过程(9.5)被称为$(p,d,q)(P,D,Q)_m$阶。常数$\theta_0$总是可以包含在(9.5)中，这会将一个确定性趋势分量引入该模型。与“非乘法”模型（9.2）比较显示$\theta(B)$和$\phi(B)$多项式已被分解为：

$\phi_{p+P}(B)=\phi_p(B)\Phi_P(B^m)$

和

$\theta_{q+Q}(B)=\theta_q(B)\Theta_Q(B^m)$



9.9由于一般乘法模型(9.5)相当复杂，很难提供其ACF和PACF的显式表达式。这使得Box和Jenkins考虑一个特别简单的案例，ARIMA（0,1,1）用于关联间隔一年的$x_t$s：

$\triangledown_mx_t=(1-\Theta B^m)\alpha_t$

类似的模型被用于关联间隔一个观察的$\alpha_t$s：

$\triangledown \alpha_t=(1-\theta B)a_t$

其中，通常$\theta$和$\Theta$具有不同的值。结合两方程，我们获得了$ARIMA(0,1,1)(0,1,1)_m$乘法模型：

$\triangledown\triangledown_mx_t=(1-\theta B)(1-\Theta B^m)\alpha_t$(9.6)

对于可逆性，我们要求$(1-\theta B)(1-\Theta B^m)$的根满足条件$|\theta|,|\Theta|<1$。模型(9.6)可以写成：

$w_t=(1-B-B^m+B^{m+1})x_t=(1-\theta B-\Theta B^m+\theta\Phi B^{m+1})a_t$

因此的自协方差$w_t$可从下式获得：

$\gamma_k=E(w_tw_{t-k})=E(a_t-\theta a_{t-1}-\Theta a_{t-m}+\theta\Theta a_{t-m-1})\times(a_{t-k}-\theta a_{t-1-k}-\Theta a_{t-m-k}+\theta\Theta a_{t-m-1-k})$

这些是

$\gamma_0=(1+\theta^2)(1+\Theta^2)\sigma^2,\gamma_1=-\theta(1+\Theta^2)\sigma^2,\gamma_{m-1}=\theta\Theta\sigma^2,\gamma_m=-\Theta(1+\theta^2)\sigma^2,\gamma_{m+1}=\theta\Theta\sigma^2$

所有其他$\gamma_k$s为零。因此，ACF为：

$\rho_1=-\frac{\theta}{1+\theta^2},\rho_{m-1}=-\frac{\theta\Theta}{(1+\theta^2)(1+\Theta^2)},\rho_m=-\frac{\Theta}{1+\Theta^2},\rho_{m+1}=\rho_{m-1}=\rho_1\rho_m$

其他$\rho_k=0$。



9.10假设模型的形式为(9.6)，则对于滞后大于m+1的样本自相关估计的方差，由下式给出：

$V(r_k)=T^{-1}(1+2(r_1^2+r_{m-1}^2+r_m^2+r_{m+1}^2)),k>m+1$(9.7)

将此结果与ACF的已知形式结合使用将使模型$ARIMA(0,1,1)(0,1,1)_m$能够识别。当然，如果SACF遵循更复杂的模式，将需要考虑$ARIMA(p,d,q)(P,D,Q)_m$其他类别的成员。



9.11 可以计算模型$ARIMA(0,1,1)(0,1,1)_m$的预测，使用7.2中概述的方法，因此：

$f_{T,h}=E(x_{T+h-1}+x_{T+h-m}-x_{T+h-m-1}+a_{T+h}-\theta a_{T+h-1}-\Theta a_{T+h-m}+\theta\Theta a_{T+h-m-1}|x_t,x_{T-1},...)$

可以显示，过程$x_t=\psi(B)a+t$的ψ权重，其中：

$\psi(B)=(1-B)^{-1}(1-B^m)^{-1}(1-\theta B)(1-\Theta B^m)$

由下式给出：

$\psi_{rm+1}=\psi_{rm+2}=...=\psi_{(r+1)m-1}=(1-\theta)(r+1-r\Theta)$

$\psi_{(r+1)m}=(1-\theta)(r+1-r\Theta)+(1-\Theta)$

这些可以用来计算（7.4）中h步提前预测的误差方差。



```python
import statsmodels.api as sm

#拟合
mod = sm.tsa.statespace.SARIMAX(df['sale'],order=(0,1,1),seasonal_order=(0,1,1,4))
results = mod.fit()
print(results.summary())
#预测
forecast = results.get_forecast(steps= 12)
f = forecast.predicted_mean
forecast_ci = forecast.conf_int()
#绘制原始序列和预测序列的时间序列图
ax = df.plot(label='observed')
forecast.predicted_mean.plot(ax=ax, label='Forecast')
ax.fill_between(forecast_ci.index,
                forecast_ci.iloc[:, 0],
                forecast_ci.iloc[:, 1], color='g', alpha=.4)
plt.legend(loc = 'upper left')
plt.show()
```

![](media/预测.png)

图9.4 啤酒销售观测，2016-2018年预测，以及95%置信区间。



对啤酒销售进行$ARIMA(0,1,1)(0,1,1)_m$建模，预测的2016-2018年的啤酒销售及95%置信区间如图9.4所示，其季节性模式十分明显。



## **季节性调整**

9.13 在2.16中，我们引入了观测时间序列的分解为趋势，季节和不规则（或噪音）分量，注意估计季节分量，然后消除它以提供一个经季节性调整的序列。扩展（8.1）中引入的符号，隐式UC分解可以写成

$x_t=z_t+s_t+u_t$(9.9)

假定附加的季节性分量$s_t$与$z_t$和$u_t$都独立。在获得季节分量的估计$\hat s_t$后，经过季节性调整的序列可以定义为$x_t^a=x_t-\hat s_t$。



9.14一个重要的问题是为什么我们希望删除季节性成分，而不是将其建模为随机过程的组成部分，例如，在拟合季节性ARIMA模型时。普遍支持的观点是能够识别，解释或反应序列中重要的非季节性运动，例如转折点和其他周期性事件，新出现的模式或突发事件，潜在原因可能会被寻找，可能会受到季节性运动的阻碍。因此，进行了季节性调整来简化数据，以便“统计小白”可以更容易解释，而不会伴随这种简化有太多的信息损失。

此限定很重要，因为它要求季节性调整程序不会导致“大量”信息丢失。尽管2.16中引入的滑动平均法很直观，并且计算简单，但它可能不是最好的可用方法。从历史上看，季节性调整方法可分为以下两种，基于经验或基于模型。移动平均法适用于前一种类别，以及统计机构开发的方法，例如美国国家局制定的人口普查程序，最新为X-13。基于模型的方法采用基于ARIMA模型的信号提取技术，拟合观测序列或其分量。



## **指数平滑**

9.15  返回两分量UC模型（8.1），其中$x_t=z_t+u_t$，那么信号或“水平”$z_t$的简单模型就是假设其当前值是$x_t$的当前和过去观测值的指数加权滑动平均值：

$z_t=\alpha x_t+\alpha(1-\alpha)x_{t-1}+\alpha(1-\alpha)^2x_{t-2}+...=\alpha\sum_{j=0}^{\infty}(1-\alpha)^jx_{t-j}$(9.10)

等式[（9.10）](https://translate.googleusercontent.com/translate_f)可以写成：

$(1-(1-\alpha)B)z_t=\alpha x_t$

或

$z_t=\alpha x_t+(1-\alpha)z_{t-1}$(9.11)

这表明当前水平$z_t$是当前观测$x_t$和过去水平$z_{t-1}$的加权平均值，权重为“平滑常数”$\alpha$。另外，(9.11)可以表示为“误差修正”形式为：

$z_t=z_{t-1}+\alpha(x_t-z_{t-1})=z_{t-1}+\alpha e_t$

这样，当前水平会从过去水平按当前误差$e_t=x_t-z_{t-1}$的比例更新，该比例再次为平滑常数α。



9.18因此，简单的指数平滑对缺少趋势的序列是一种合适的预测程序。为了捕捉线性趋势，可以通过扩展（9.11）以包括趋势成分来更新该方法，

$z_t=\alpha x_t+(1-\alpha)(z_{t-1}+\tau_{t-1})=z_{t-1}+\tau_{t-1}+\alpha e_t$(9.12)

其中误差修正现在为$e_t=x_t-z_{t-1}-\tau_{t-1}$，并且定义了趋势的第二更新方程：

$\tau_t=\beta(z_t-z_{t-1})+(1-\beta)\tau_{t-1}=\tau_{t-1}+\alpha\beta e_t$(9.13)

这对更新方程共同称为Holt-Winters模型。预测如下：

$f_{T,h}=z_T+\tau_Th$(9.14)

因此，位于“局部”线性趋势上，其截距和斜率在每个期间由等式(9.12)和（9.13）更新。使用这些循环关系，可以看出，Holt-Winters模型等价于ARIMA（0，2，2）过程：

$\triangledown^2x_t=(1-(2-\alpha-\alpha\beta)B-(\beta-1)B^2)a_t$

因此，就一般过程而言

$\triangledown^2x_t=(1-\theta_1B-\theta_2)a_t$(9.15)

平滑参数通过$\alpha=1+\theta_2$和$\beta=(1-\theta_1-\theta_2)/(1+\theta_2)$给出。注意当α=β=0时，$\tau_t=\tau_{t-1}=...=\tau$，$z_t=z_{t-1}+\tau=z_0+\tau t$，其中$z_0$是水平分量的初始值。然后通过“全局”线性趋势$f_{T,h}=z_0+\tau (T+h)$给出预测。此外，在这种情况下，$\theta_1=2,\theta_2=-1$，因此：

$\triangledown^2x_t=(1-2B+B^2)a_t=\triangledown^2a_t$

等效于$x_t$的趋势平稳（TS）模型。



9.20 Holt-Winters框架很容易适应季节性。基于(9.9)，加法Holt-Winters水平更新方程（9.12）为

$z_t=\alpha (x_t-s_{t-m})+(1-\alpha)(z_{t-1}+\tau_{t-1})=z_{t-1}+\tau_{t-1}+\alpha (x_t-s_{t-m}-z_{t-1}-\tau_{t-1})=z_{t-1}+\tau_{t-1}+\alpha e_t$

趋势更新方程仍为（9.13），还有一个附加的季节更新方程

$s_t=\delta(x_t-z_t)+(1-\delta)s_{t-m}=s_{t-m}+\delta(1-\beta)e_t$

预测为

$f_{T,h}=z_T+\tau_T+s_{T+h-m}$

这些更新方程式可以证明是与ARIMA模型等价（Newbold，1988）：

$\triangledown\triangledown_m x_t=\theta_{m+1}(B)a_t$(9.16)

其中

$\theta_1=1-\alpha-\alpha\beta,\theta_2=...=\theta_{m-1}=-\alpha\beta,\theta_m=1-\alpha\beta-(1-\alpha)\delta,\theta_{m+1}=-(1-\alpha)(1-\delta)$

如果β=0，则趋势是恒定的，如果$\theta_!\theta_m+\theta_{m+1}=0$，或等价地$2-2\delta+\alpha\delta=0$，(9.16)降低为$ARIMA(0,1,1)(0,1,1)_m$模型。



9.21在对季节时间序列建模时，可能会认为（9.9）的加法分解是不合适的，通常认为季节运动与水平成比例，噪声分量仍会相加，即分解形式为：

$x_t=z_ts_t+u_t$

当然，排除使用需要所有分量相乘的对数变换的情况。在这种情况下，可以使用乘法Holt-Winters模型，其更新公式为：

$z_t=\alpha(\frac{x_t}{s_{t-m}})+(1-\beta)(z_{t-1}+\tau_{t-1})=z_{t-1}+\tau_{t-1}+\frac{\alpha e_t}{s_{t-m}}$

$\tau_t=\beta(z_t-z_{t-1})+(1-\beta)\tau_{t-1}=\tau_{t-1}+\frac{\alpha\beta e_t}{s_{t-m}}$

$s_t=\delta(\frac{x_t}{z_t})+(1-\delta)s_{t-m}=s_{t-m}+\delta(1-\alpha)\frac{e_t}{z_t}$

预测为：

$f_{T,h}=(z_T+\tau_Th)s_{T+h-m}$

该模型似乎没有等效的ARIMA表示。注意，在加法和乘法Holt-Winters模型中都设置δ=0，无法消除季节性分量，它可以将季节因子限制为常数，既然$s_t=s_{t-m}$。

