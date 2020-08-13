# **Chapter 4 非平稳时间序列的ARMA模型**

## **非平稳**

4.1 自回归滑动平均（ARMA）一类的模型依赖于假设潜在的过程是弱平稳的，将均值和方差限制为常数，并要求自相关仅依赖于时滞。但是，正如我们所看到的，很多时间序列肯定不是平稳的，因为它们往往表现出随时间变化的均值和/或方差。



4.2为了解决这种非平稳的情况，我们首先描述一个时间序列是非常数的平均水平加随机误差的总和：

$x_t=\mu_t+\varepsilon_t$(4.1)

非平稳平均水平$\mu_t$可以用多种方法建模。一种潜在的现实可能性是，均值为（非随机的）时间d阶的多项式，误差$\varepsilon_t$假定成为随机的，平稳的但可能自相关的零均值过程。实际上，考虑到Cramer（1961）对Wold

分解定理扩展到非平稳过程。因此，我们可能有：

$x_t=\mu_t+\varepsilon_t=\sum_{j=0}^d\beta_jt^j+\psi(B)a_t$(4.2)

既然：

$E(\varepsilon_t)=\psi(B)E(a_t)=0$

我们有

$E(x_t)=E(\mu_t)=\sum_{j=0}^d\beta_jt^j$

并且，由于系数$\beta_j$保持恒定，在这样一种趋势下均值是确定性的。



4.3这种趋势可以通过简单的转换来消除。考虑通过设置d=1获得的线性趋势，为简单起见，其中误差分量假定为白噪声序列：

$x_t=\beta_0+\beta_1t+a_t$(4.3)

一周期滞后$x_t=\beta_0+\beta_1t+a_t$(4.3)为：

$x_{t-1}=\beta_0+\beta_1(t-1)+a_{t-1}=\beta_0-\beta_1+\beta_1t+a_{t-1}$

并从(4.3)减去得到：

$x_t-x_{t-1}=\beta_1+a_t-a_{t-1}$（4.4）

结果是遵循ARMA（1,1）过程的差分方程，其中，既然$\phi=\theta=1$，自回归和滑动平均根都是1，模型既不平稳也不可逆。但是，如果我们考虑一阶微分$w_t=\bigtriangledown x_t$，那么（4.4）可以被写为：

$w_t=\beta_1+\bigtriangledown a_t$

既然$E(w_t)=\beta_1$是常数，因此$w_t$由平稳，但不可逆的MA（1）过程产生。



4.4一般情况下，如果趋势多项式为d阶多项式且$\varepsilon_t$具有ARMA过程$\phi(B)\varepsilon_t=\theta(B)a_t$的特征，则$\bigtriangledown^dx_t=(1-B)^dx_t$，由d次一阶微分$x_t$得到，将遵循以下过程：

$\bigtriangledown^dx_t=\theta_0+\frac{\bigtriangledown^d\theta(B)}{\phi(B)}a_t$

其中$\theta_0=d!\beta_d$。因此，过程的滑动平均（MA）部分生成的$\bigtriangledown^dx_t$也将包含因子$\bigtriangledown^d$,因此，将有d个单位根。注意，$x_t$的方差将与$\varepsilon_t$的方差相同，因此对于所有t都是恒定的。图4.1显示了由线性和二次（d=2）趋势模型生成的数据的图。因为误差分量的方差，这里假定为白噪声，且服从NID（0，9），是恒定的并且与水平独立，每个序列的可变性以期望值为界，趋势分量在图中能清楚地观察到。

```python
# 模拟的线性和二次趋势
a = [i for i in range(0,101)]
x1 = [0]
x2 = [0]
for i in range(1,101):
    n = random.gauss(0, 9)
    m1 = 10 + 2*i + n
    m2 = 10 + 5*i -0.03*(i**2) + n
    x1.append(m1)
    x2.append(m2)

plt.xlabel("t")
plt.ylabel("x_t")
plt.plot(a,x1,label='M1')
plt.plot(a,x2,label='M2')
plt.legend()
plt.show()
```

![趋势](media/趋势.png)

图4.1模拟的线性和二次趋势。



4.5生成非平稳平均水平的另一种方法是使用自回归参数不满足平稳条件的ARMA模型。例如，考虑AR（1）过程：

$x_t=\phi x_{t-1}+a_t$(4.5)

其中$\phi>1$。如果假定该过程在时间t=0开始，则差分方程(4.5)有解：

$x_t=x_0\phi^t+\sum_{i=0}^t\phi^i a_{t-i}$(4.6)

“补充函数”$x_0\phi^t$可以认为是$x_t$在时间t=0的条件期望，显然为t的增函数。$x_t$在时间t=0,1,2...的条件期望将取决于随机冲击序列$a_1,a_2,...$，因此，由于这种期望条件可被视为$x_t$的趋势，趋势随机变化。

$x_t$的方差由下式给出：

$V(x_t)=\sigma^2\frac{\phi^{2t}-1}{\phi^2-1}$

这也是时间的增函数，且随着$t\to \infty$趋于无穷。通常，$x_t$的均值和方差都有趋势，且过程是爆炸性的。从过程(4.5)生成的数据的图，其中$\phi=1.05$，$a_t\sim NID(0,9)$，初始值$x_0=10$，如图4.2所示。我们看到，在短暂的“感应期”之后，该序列基本上遵循指数曲线，生成的$a_t$们几乎没有进一步起作用。如果将自回归和滑动平均项添加到模型中，将观察到相同的行为，只要违反平稳性条件。

```python
# 模拟的爆炸的AR（1）模型
random.seed(20)
a = [i for i in range(0,76)]
x = [10]
for i in range(1,76):
    n = random.gauss(0, 9)
    r= 1.05*x[i-1] + n
    x.append(r)
    
plt.xlabel("t")
plt.ylabel("x_t")
plt.plot(a,x)
plt.show()
```

![ar1](media/ar1.png)

图4.2模拟的爆炸的AR（1）模型。



## **ARIMA过程**

4.6从(4.6)可以看到，如果$\phi>1$，则(4.5)的解是爆炸性的;但如果$\phi<1$，则平稳。情况$\phi=1$产生的过程在两者之间完全平衡。如果$x_t$由模型生成：

$x_t=x_{t-1}+a_t$（4.7）

则$x_t$遵循随机游走。如果我们让一个常数$\theta_0$，包括在内，因此有：

$x_t=x_{t-1}+\theta_0+a_t$（4.8）

则$x_t$将遵循有漂移的随机游走。如果过程从t=0开始，则有:

$x_t=x_0+t\theta_0+\sum_{i=0}^ta_{t-i}$

因此有：

$\mu_t=E(x_t)=x_0+t\theta_0$

$\gamma_{0,t}=V(x_t)=t\sigma^2$

且

$\gamma_{k,t}=Cov(x_t,x_{t-k})=(t-k)\sigma^2,k\ge0$

都是t的函数，因此是随时间变化的。



4.7 $x_t$和$x_{t-k}$之间的自相关如下：

$\rho_{k,t}=\frac{\gamma_{k,t}}{\sqrt{\gamma_{0,t},\gamma_{0,t-k}}}=\frac{t-k}{\sqrt {t(t-k)}}=\sqrt{\frac{t-k}{t}}$

如果t与k相比较大，则所有$\rho_{k,t}$将大约为一。因此，$x_t$值的序列将非常平滑，但也将是非平稳的，因为$x_t$的均值和方差都将随着t的变化而变化。图4.3显示出，$x_0=10,a_t\sim NID(0,9)$所产生的随机游动的曲线（4.7）和（4.8）。在该图(A)中的漂移参数，$\theta_0$，设定为零，而在（B）设置$\theta_0=2$。两个曲线非常不同，但与初始值都没有任何关系：实际上，随机行走再次通过任意值的期望时间长度是无限的。

```python
#模拟的随机游动
random.seed(4)
a = [i for i in range(0,101)]
x1 = [10]
x2 = [10]
for i in range(1,101):
    n = random.gauss(0, 9)
    r1= x1[i-1] + n
    r2= x2[i-1] + n + 2
    x1.append(r1)
    x2.append(r2)
    
#设置画布
plt.figure(figsize=(15,5))

plt.subplot(121)
plt.xlabel("t")
plt.ylabel("x_t")
plt.plot(a,x1)
plt.title('(A)')

plt.subplot(122)
plt.xlabel("t")
plt.ylabel("x_t")
plt.plot(a,x2)
plt.title('(B)')
plt.show()
```

![随机游走](media/随机游走.png)

图4.3模拟的随机游动 ，$(A)x_t=x_{t-1}+a_t,x_0=10,a_t\sim NID(0,9)$,$(B)x_t=2+x_{t-1}+a_t,x_0=10,a_t\sim NID(0,9)$



4.8随机游走是一类非平稳模型的例子，被称为单积过程。等式(4.8)可以写成：

$\bigtriangledown x_t=\theta_0+a_t$

所以$x_T$的一阶差分导出一个平稳模型，在这种情况下，白噪声过程$a_t$。通常，一个序列可能需要d次一阶差分以达到平稳性，这样获得的序列本身可能是自相关的。

如果此自相关是由ARMA（p，q）过程建模的，则原始序列的模型具有以下形式：

$\phi(B)\bigtriangledown^dx_t=\theta_0+\phi(B)a_t$(4.9)

称为p，d和q阶的单积自回归滑动平均（ARIMA）过程或ARIMA（p，d，q），且$x_t$为d阶单积，表示为$I(d)$。



4.9通常情况下，单积阶d，或等价的，差分阶，将为0或1，偶尔为2。对于所有不大的k，ARIMA过程的自相关值将接近1。例如，考虑（平稳的）ARMA（1,1）过程：

$x_t-\phi x_{t-1}=a_t-\theta a_{t-1}$

其ACF为

$\rho_1=\frac{(1-\phi \theta)(\phi-\theta)}{1+\theta^2-2\phi \theta},\rho_k=\phi\rho_{k-1},k>1$

当$\phi\to\infty$时，ARIMA（0,1,1）过程：

$\bigtriangledown x_t=a_t-\theta_0a_{t-1}$

结果，所有的$\rho_k$趋于1。



4.10有关ARIMA类型的模型的几点很重要。再考虑一次(4.9)，为简单起见$\theta_0=0$：

$\phi(B)\bigtriangledown^dx_t=\theta(B)a_t$

该过程可以等效地由两个方程式定义：

$\phi(B)w_t=\theta(B)a_t$

和

$w_t=\bigtriangledown^dx_t$(4.10)

因此，如前所述，模型对应于假设$\bigtriangledown^dx_t$可以用平稳、可逆的ARMA过程表示。或者，对于，可以将(4.10)取逆得到：

$x_t=S^dw_t$(4.11)

其中S是由下式定义的无限求和或单积算子

$S=(1+B+B^2+...)=(1-B)^{-1}=\bigtriangledown^{-1}$

等式(4.11)显示可以通过求和或“单积”平稳序列$w_t$d次得到，因此，为单积过程。



4.11这种非平稳行为通常被称为齐次非平稳性，重要的是讨论为什么这种形式的非平稳性在描述许多领域的这种行为的时间序列时非常有用。再次考虑一阶自回归过程(4.2)。一个AR（1）模型的基本特征是，对于$|\phi|<1$和$\phi>1$,由这模型生成的序列的“局部”行为在很大程度上取决于$x_t$的水平。在前一种情况下，局部行为将永远是由均值的吸引力控制，而在后一种情况下，该序列将最终随着t迅速增加。但是，对于许多时间序列，局部行为似乎与水平无关，这就是我们说齐次非平稳性的意思。



4.12如果我们要使用ARMA模型，其过程的行为确实与它的水平无关，那么必须选择自回归多项式$\phi(B)$，因此：

$\phi(B)(x_t+c)=\phi(B)x_t$

其中c是任何常数，从而：

$\phi(B)c=0$

这意味着$\phi(1)=0$，因此$\phi(B)必$须能够被因式分解为

$\phi(B)=\phi_1(B)(1-B)=\phi_1(B)\bigtriangledown$

在这种情况下，需要考虑的过程类别将是

$\phi_1(B)w_t=\theta(B)a_t$

其中$w_t=\bigtriangledown x_t$。既然齐次非平稳性的需要阻碍了$w_t$爆炸增长，则$\phi_1(B)$是平稳算子或$\phi_1(B)=\phi_2(B)(1-B)$，从而$\phi_2(B)w^*_t=\theta(B)a_t$，其中$w^*_t=\bigtriangledown^2x_t$。既然参数可以递归使用，因此对于齐次非平稳时间序列，自回归滞后多项式必须是$\phi(B)\bigtriangledown^d$的形式，其中$\phi(B)$是平稳多项式。图4.4曲线图的数据来自模型$\bigtriangledown^dx_t=a_t$，其中，$a_t\sim NID(0,9),x_0=x_1=10$，并且可以看到这样的序列在水平和斜率方向上都显示出随机运动。

```python
# 模拟的“二次差分”模型
random.seed(14)
a = [i for i in range(0,101)]
x = [10,10]
for i in range(2,101):
    n = random.gauss(0, 9)
    r= -x[i-2] + 2*x[i-1] + n
    x.append(r)
    
plt.xlabel("t")
plt.ylabel("x_t")
plt.plot(a,x)
plt.show()
```

![二次差分](media/二次差分.png)

图4.4模拟的“二次差分”模型。

## 

4.13一般来说，如果一个常数包含在d次差分的模型中，那么会自动允许自由度为d的确定的多项式趋势。同样地，如果$\theta_0$取为非零，则：

$E(w_t)=E(\bigtriangledown^dx_t)=\mu_w=\frac{\theta_0}{(1-\phi_1-\phi_2-...-\phi_p)}$

非零，因此表示(4.9)的另一种方式是

$\phi(B)\widetilde w_t=\theta(B)a_t$

其中$\widetilde w_t=w_t-\mu_w$。

图4.5绘制了$\bigtriangledown^2x_t=2+a_t$生成的数据，其中，$a_t\sim NID(0,9),x_0=x_1=10$。确定的二次趋势对序列的演变有重大影响，一段时间后，“噪音”被完全淹没。

因此，模型(4.9)允许模拟随机趋势和确定性趋势。当$\theta_0=0$时,只有一个随机趋势被合并，而如果$\theta_0\ne 0$，模型可以解释为代表一个确定性趋势（一个d阶多项式）埋在非平稳和自相关噪声中，后者包含随机趋势。4.2‑4.4中介绍的模型可以说是埋在平稳噪声中的确定性趋势，因为它们可以写成：

$\phi(B)\bigtriangledown^dx_t=\phi(1)\beta_dd!+\bigtriangledown^d\theta(B)a_t$

滑动平均滞后多项式的d个根为1，证明$x_t$水平的噪声的平稳性质。

```python
#模拟的“带漂移的二次差分”模型
random.seed(14)
a = [i for i in range(0,101)]
x = [10,10]
for i in range(2,101):
    n = random.gauss(0, 9)
    r= -x[i-2] + 2*x[i-1] + n +2
    x.append(r)
    
plt.xlabel("t")
plt.ylabel("x_t")
plt.plot(a,x)
plt.show()
```

![带漂移](media/带漂移.png)

图4.5模拟的“带漂移的二次差分”模型。



## **ARIMA模型**

4.14一旦确定了差分次数d，根据定义$w_t=\bigtriangledown^dx_t$，是平稳的，3.29‑3.35中讨论的ARMA模型构建技术可能适用于适当差分的序列。建立正确的差分次数绝非易事，将在5.4-5.7中进行详细讨论。