# **Chapter 3 平稳时间序列的ARMA模型**

## **随机过程和稳定性**

3.1 有必要更严格地考虑平稳时间序列的概念。为此，通常视序列$x_t$上的观测值$x_1,x_2,...,x_T$为一次随机过程的实现。通常这个随机过程通过T维的概率分布来描述，使一次实现和随机过程的关系，在经典统计中类似于，样本与从中提取样本的总体的关系。

但是，指定概率分布的完整形式，通常是过于雄心勃勃的任务，因此注意力通常集中在一阶和二阶矩；T个均值：

$E(x_1),E(x_2),...,E(x_T)$

T个方差：

$V(x_1),V(x_2),...,V(x_T)$

和$T(T-1)/2$个协方差：

$Cov(x_i,x_j),i<j$

如果可以认为分布是（多元）正态分布，那么期望的集合将完全刻画出随机过程的特征。这样的假设并不总是合适的，但如果过程是线性的，则当前值$x_t$由过程的本身的先前值$x_{t-1},x_{t-2},...$的线性组合加上任何其他相关过程的当前值和过去值确定，则期望的集合将再次抓住它的主要特性。

但是，无论哪种情况，都无法仅从过程的一次实现推断出所有值的一阶和二阶矩，因为只有T个观测值，而有$T+T(T+1)/2$个未知参数。因此，必须做出进一步简化的假设，减少未知参数的数量，这样有更可控的性质。



3.2应该强调的是，使用单一实现的程序推断联合概率分布的未知参数,仅在过程是遍历的时有效，这大致意味着有限次实现的样本矩瞬间接近它们的总体对应的实现，当实现的长度变得无限时。既然很难仅使用一次（部分）实现的来测试遍历性，这将假定此属性在每个时间序列中均成立。



3.3 在第一章中介绍了最重要的平稳性简化假设，要求过程处于“统计平衡”状态。如果随机过程具有以下特点，则称其为严格平稳的：不受时间起点变化的影响，即任何时间$t_1,t_2,...,t_m$的联合概率分布必须与$t_{1+k},t_{2+k},...,t_{m+k}$的联合概率分布相同;其中k是任意时移。对于m=1，严格平稳性意味着在$t_1,t_2,...$的边际概率分布不依赖时间，这反过来意味着只要$E|x_t^2|<\infty$是有限的第二时刻假设的一部分），$x_t$的均值和方差肯定是常数，因此：

$E(x_1),E(x_2),...,E(x_T)=\mu$

和

$V(x_1),V(x_2),...,V(x_T)=\sigma_x^2$

如果m=2，严格平稳性意味着所有二元分布不取决于时间，因此协方差是时移（或滞后）k的函数，因此对所有k，

$Cov(x_1,x_{1+k})=Cov(x_2,x_{2+k})=...=Cov(x_{T-k},x_T)=Cov(x_t,x_{t-k})$

这导致了lag-k自协方差的定义为：

$\gamma_k=Cov(x_t,x_{t-k})=E((x_t-\mu)(x_{t-k}-\mu))$

因此

$\gamma_0=E(x_t-\mu)^2=V(x_t)=\sigma_x^2$

然后可以将lag-k自相关定义为

$\rho_k=\frac{Cov(x_t,x_{t-k})}{(V(x_t)V(x_{t-k}))^{1/2}}=\frac{\gamma_k}{\gamma_0}=\frac{\gamma_k}{\sigma_x^2}$(3.1)

$x_t$的均值和方差均为常数，自协方差和自相关仅取决于滞后k的假设集被称为弱平稳性或者协方差平稳性。



3.4 虽然严格的平稳性（具有有限的二阶矩）暗示弱平稳性，反之则不成立，有可能一个过程是弱平稳的，但不严格平稳。例如更高的矩$E(x_t^3)$，是时间的函数。但是，如果可以假定联合概率被分布的前两个矩完全刻画，弱平稳确实意味着严平稳。



3.5 当考虑作为k的函数时，自相关集合（3.1）称为（总体）自相关函数（ACF）。既然：

$\gamma_k=Cov(x_t,x_{t-k})=Cov(x_{t-k},x_t)=\gamma_{-k}$

它遵循$\rho_{-k}=\rho_k$，因此通常只要给定ACF的正半部分。ACF在$x_t$之间依赖关系的建模中起主要作用，表征过程的均值$\mu=E(x_t)$以及方差$\sigma_x^2=\gamma_0=V(x_t)$，平稳随机过程描述$x_t$的演化。因此表明，它通过测量一个过程的值与先前的值，长度和过程“记忆”的强度相关程度。



## **WOLD分解与自相关**

3.6时间序列分析中的一个基本定理，称为Wold分解，指出每个弱平稳，纯粹不确定的随机过程$x_t-\mu$可以写为一个不相关的随机变量序列的线性组合（或线性滤波器）。“完全不确定”是指任何确定性分量已经从$x_t-\mu$减去。这样的部分是可以根据它们过去的值完美预测的，常见的例子是（常数）平均值，如过程写为$x_t-\mu$，周期性的序列（例如，正弦和余弦函数），以及t的多项式或指数序列。

该线性滤波器表示为：

$x_t-\mu=a_t+\psi_1a_{t-1}+\psi_2a_{t-2}+...=\sum_{j=0}^\infty\psi_ja_{t-j},\psi_0=1$(3.2)

$a_t,t=0,\pm1,\pm2,...$是一系列不相关的随机变量，通常称为新息，来自具有以下特征的固定分布：

$E(a_t)=0,V(a_t)=E(a_t^2)=\sigma^2<\infty$

和

$Cov(a_t,a_{t-k})=E(a_ta_{t-k})=0,k\ne0$

这样的序列称为白噪声过程，新息有时会表示为$a_t\sim WN(0,\sigma^2)$。线性滤波器(3.2)中的系数（可能的无穷大）称为ψ权重。



3.7容易表明，模型(3.2)导致$x_t$的自相关。从该等式可以得出：

$E(x+t)=\mu$

和

$\gamma_0=\sigma^2\sum_{j=0}^\infty\psi_j^2$

通过使用白噪声结果$E(a_{t-i}a_{t-j})=0,i\ne j$ ，有：

$\gamma_k=\sigma^2\sum_{j=0}^\infty\psi_j\psi_{j+k}$

这意味着

$\rho_k=\frac{\sum_{j=0}^\infty\psi_j\psi_{j+k}}{\sum_{j=0}^\infty\psi_j^2}$

如果(3.2)的ψ权重有无穷个，权重必须被假定为绝对可加，因此$\sum_{j=0}^\infty|\psi_j|<\infty$，在这种情况下线性滤波器被认为是收敛的。这个条件可以证明与以下假设是等效的：假设$x_t$是平稳的，并保证所有矩都存在且时间独立，特别的，$x_t,\gamma_0$的方差是有限的。



## **一阶自回归过程**

3.8虽然等式（3.2）看起来可能复杂，许多现实模型的结果由特定选择的ψ权重得出。不损失一般性取μ=0，选择$\psi_j=\phi^j$使（3.2）被写为：

$x_t=a_t+\phi a_{t-1}+\phi^2a_{t-2}+...=\phi x_{t-1}+a_t$

或者

$x_t-\phi x_{t-1}=a_t$（3.3）

这被称为一阶自回归过程，通常被称为AR（1）。



3.9在2.10中引入的滞后算子$B$允许（可能是无限的）滞后表达方式简洁明了。例如，可以将AR（1）过程写为：

$(1-\phi B)x_t=a_t$

因此

$x_t=(1-\phi B)^{-1}a_t=(1+\phi B+\phi^2B^2+...)a_t=a_t+\phi a_{t-1}+\phi^2a_{t-2}+...$(3.4)

如果$|\phi|<1$，则该线性滤波器表示将收敛，因此，为平稳条件。



3.10现在可以推导出AR（1）过程的ACF。（3.3）的两边乘以$x_{t-k},k>0$，取期望：

$\gamma_k-\phi\gamma_{k-1}=E(a_tx_{t-k})$(3.5)

从(3.4)，$a_tx_{t-k}=\sum_{i=0}^\infty\phi^ia_ta_{t-k-i}$。由于$a_t$是白噪声，因此$a_ta_{t-k-i}$的任意一项期望为零，对于所有$k+i>0$。因此（3.5）可以简化为：

$\gamma_k=\phi\gamma_{k-1},k>0$

因此$\gamma_k=\phi^k\gamma_0$，AR（1）过程具有ACF $\rho_k=\phi^k$。因此，如果$\phi>0$，ACF呈指数衰减至零，而如果$\phi<0$则ACF呈振荡模式衰减，如果$\phi$接近+1和-1的非平稳边界,则均衰减地很慢。



3.11两个AR（1）过程的ACF，（A）$\phi=0.5$和（B）$\phi=-0.5$，如图所3.1所示，以及假定$a_t$为正态和独立分布$\sigma^2=25$的两个过程产生的数据，记$a_t\sim NID(0,25)$，且起始值为$x_0=0$（通常$a_t$是正态白噪声，因为在正态情况下独立性意味着不相关）。$\phi>0$,$x_t$的相邻值呈正相关，并且生成的序列趋于平滑，表现出具有相同的符号。但是，当$\phi<0$时，相邻值呈负相关，并且生成的序列显示剧烈，快速的振荡。

```python
# AR（1）过程的ACF
a = [i for i in range(1,13)]
b = [1]
for i in range(len(a)):
    n = b[i]*0.5
    b.append(n)
del(b[0])
#设置画布
plt.figure(figsize=(10,10))

plt.subplot(221)
plt.xlabel("k")
plt.ylabel("ρ_k")
plt.ylim([-1,1])
plt.bar(a,b, width=0.1)
plt.axhline(y=0)
plt.title('(A)')

b = [1]
for i in range(len(a)):
    n = b[i]*-0.5
    b.append(n)
del(b[0])

plt.subplot(222)
plt.xlabel("k")
plt.ylabel("ρ_k")
plt.ylim([-1,1])
plt.bar(a,b, width=0.1)
plt.axhline(y=0)
plt.title('(B)')

# AR（1）过程的模拟
a = [i for i in range(0,101)]
x1 = [0]
x2 = [0]
for i in range(len(a)-1):
    n = random.gauss(0, 25)
    a_t1 = n + 0.5*x1[i]
    a_t2 = n - 0.5*x1[i]
    x1.append(a_t1)
    x2.append(a_t2)
    
plt.subplot(223)
plt.plot(a,x1)
plt.xlabel("t")
plt.ylabel("x_t")
plt.title('(C)')

plt.subplot(224)
plt.plot(a,x2)
plt.xlabel("t")
plt.ylabel("x_t")
plt.title('(D)')
plt.show()
```

![ar1](media/ar1.png)

图3.1 AR（1）过程的 ACF（A）$\phi=0.5$（B）$\phi=-0.5$

和模拟（C）$\phi=0.5$，$x_0=0$（D）$\phi=-0.5$，$x_0=0$  。ACF，自相关函数。



## **一阶滑动平均过程**

3.12现在考虑通过选择$\psi_1=-\theta,\psi_j=0,j\ge2$获得的模型，在（3.2）中：

$x_t=a_t-\theta a_{t-1}$

或

$x_t=(1-\theta B)a_t$

这就是一阶滑动平均（MA（1））过程，它遵循：

$\gamma_0=\sigma^2(1+\theta^2),\gamma_1=-\sigma^2\theta,\gamma_k=0,k>1$

因此，其ACF由下式描述：

$\rho_1=-\frac{\theta}{1+\theta^2},\rho_k=0,k>1$

因此，尽管相隔一个周期的观测是相关的，但是间隔超过一个周期的观测不相关，因此该过程的记忆仅仅是一个周期：在k=2处“跳变”到零自相关可能与AR（1）过程的ACF的平滑、指数衰减形成对比。



3.13 $\rho_1$的表达式可以写为二次方程$\rho_1\theta^2+\theta+\rho_1=0$。由于$\theta$必须是实数，它遵循$|\rho_1|<0.5$。但是$\theta$，$1/\theta$都满足该方程，因此，对于相同的ACF，总可以发现两个对应的MA（1）过程。



3.14 既然任何MA模型都包含有限数量的ψ权重，因此所有MA模型是平稳的。然而为了获得收敛的自回归表示，必须施加限制$\theta<1$。这个限制称为可逆性条件，表示该过程可以写成无限的自回归表示形式：

$x_t=\pi_1 x_{t-1}+\pi_2 x_{t-2}+...+a_t$

其中π权重收敛：$\sum_{j=1}^\infty |\pi_j|<\infty$。实际上，MA（1）模型可以被写成：

$(1-\theta B)^{-1}x_t=a_t$

将$(1-\theta B)^{-1}$展开得到

$(1+\theta B+\theta^2 B^2+...)x_t=a_t$

如果$|\theta|<1$，换句话说，如果模型是可逆的，权重$\pi_j=\theta^j$将收敛。这暗示着合理的假设，即过去观测的影响与日俱减。



3.15 图3.2展示了两个MA（1）过程生成的数据图，分别是（A）θ=0.8和（B）θ=-0.8，每个例子中，都有$a_t\sim NID(0,25)$。将这些图和图3.1中AR（1）过程做比较，可以看到，这两个过程的实现往往是相当类似的（例如在$\rho_1$的值分别为0.488和0.5），因此有时可能难以区分两者。

```python
# MA（1）过程的模拟
a = [i for i in range(0,101)]
x0 = random.gauss(0, 25)
x1 = [x0]
x2 = [x0]
r = [x0]
for i in range(len(a)-1):
    n = random.gauss(0, 25)
    r.append(n)
    a_t1 = n - 0.8*r[i]
    a_t2 = n + 0.8*r[i]
    x1.append(a_t1)
    x2.append(a_t2)

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

![ma1](media/ma1.png)

图3.2 MA（1）过程的模拟。（A）θ=0.8和（B）θ=-0.8。MA，滑动平均。



## **一般的AR和MA过程**

3.16 直接扩展AR（1）和MA（1）模型。一般的p阶的自回归模型（AR（p））可以写成：

$x_t-\phi_1x_{t-1}-\phi_2x_{t-2}-...-\phi_px_{t-p}=a_t$

或

$(1-\phi_1B-\phi_2B^2-...-\phi_pB^p)x_t=\phi (B)x_t=a_t$

线性滤波器$x_t=\phi^{-1} (B) a_t=\psi(B)a_t$表示可通过在$\phi(B)\psi(B)=1$等同系数获得。



3.17 ψ权重收敛的平稳条件是特征方程的根：

$\phi(B)=(1-g_1B)(1-g_2B)...(1-g_pB)=0$

其中$|g_i|<1,i=1,2,...,p$。ACF的行为由差分方程决定：

$\phi(B)\rho_k=0,k>0$（3.7）

其解为：

$\rho_k=A_1g_1^k+A_2g_2^k+...+A_pg_p^k$

既然$|g_i|<1$，因此，ACF是通过衰减指数（实根）和阻尼正弦波（复根）的混合来描述的。例如，考虑AR（2）过程：

$(1-\phi_1B-\phi_2B^2)x_t=a_t$

具有特征方程

$\phi(B)=(1-g_1B)(1-g_2B)=0$

根$g_1$和$g_2$由下式给出：

$g_1,g_2=(1/2)(\phi_1\pm(\phi_1^2+4\phi_2)^{1/2})$

并且可以都是实数，也可以是一对复数。由于平稳性，要求根$|g_1|,|g_2|<1$，这些条件意味着在$\phi_!$和$\phi_2$上的一组限制：

$\phi_!+\phi_2<1,-\phi_!+\phi_2<1,-1<\phi_2<1$

如果$\phi_1^2+4\phi_2<0$，根将是复数，虽然根是复数一个必要条件是简单的$\phi_2<0$。



3.18 AR（2）过程的ACF由以下公式给出：

$\rho_k=\phi_1\rho_{k-1}+\phi_2\rho_{k-2}$

对于$k\ge2$及初始值$\rho_0=1,\rho_1=\phi_1/(1-\phi_2)$。此ACF的$(\phi_1,\phi_2)$四种组合的行为如图3.3所示。如果$g_1$和$g_2$是实数（情况（A）和（C）），则ACF是两个衰减指数的混合。根据它们的符号，自相关也可以衰减为振荡的方式。如果根是复数（情况（B）和（D）），则ACF遵循衰减正弦波。图3.4显示了这四个AR（2）过程的生成的时间序列图，每种情况下，$a_t\sim NID(0,25)$。根据实根的符号，序列可能是平滑的，也可能是锯齿状，而复根往往会诱发“周期性”行为。

```python
#各种AR（2）过程的ACF
a = [i for i in range(1,13)]

def ACFs(a1,a2):
    b = [1,a1/(1-a2)]
    for i in range(11):
        n = a1*b[i+1] + a2*b[i]
        b.append(n)
    return b
# (A)
b = ACFs(0.5,0.3)
del(b[0])
#设置画布
plt.figure(figsize=(10,10))

plt.subplot(221)
plt.xlabel("k")
plt.ylabel("ρ_k")
plt.ylim([-1,1])
plt.bar(a,b, width=0.1)
plt.axhline(y=0)
plt.title('(A)')
# (B)
b = ACFs(1,-0.5)
del(b[0])

plt.subplot(222)
plt.xlabel("k")
plt.ylabel("ρ_k")
plt.ylim([-1,1])
plt.bar(a,b, width=0.1)
plt.axhline(y=0)
plt.title('(B)')
# (C)
b = ACFs(-0.5,0.3)
del(b[0])

plt.subplot(223)
plt.xlabel("k")
plt.ylabel("ρ_k")
plt.ylim([-1,1])
plt.bar(a,b, width=0.1)
plt.axhline(y=0)
plt.title('(C)')
# (D)
b = ACFs(-0.5,-0.3)
del(b[0])

plt.subplot(224)
plt.xlabel("k")
plt.ylabel("ρ_k")
plt.ylim([-1,1])
plt.bar(a,b, width=0.1)
plt.axhline(y=0)
plt.title('(D)')
```

![ar2](media/ar2.png)

图3.3各种AR（2）过程的ACF，$(A)\phi_1=0.5,\phi_2=0.3$，$(B)\phi_1=1,\phi_2=-0.5$，$(C)\phi_1=-0.5,\phi_2=0.3$，$(D)\phi_1=-0.5,\phi_2=-0.3$。ACF，自相关函数。

```python
#各种AR（2）过程的模拟
a = [i for i in range(0,101)]

def ACFs(a1,a2,ran):
    x = [0,0]
    for i in range(99):
        n = a1*x[i+1] + a2*x[i] + ran[i]
        x.append(n)
    return x

ran = []
for i in range(99):
        ran.append(random.gauss(0, 25))
x1 = ACFs(0.5,0.3,ran)
x2 = ACFs(1,-0.5,ran)
# (A) 
#设置画布
plt.figure(figsize=(15,5))

plt.subplot(121)
plt.xlabel("t")
plt.ylabel("x_t")
plt.plot(a,x1)
plt.title('(A)')
# (B)
plt.subplot(122)
plt.xlabel("t")
plt.ylabel("x_t")
plt.plot(a,x2)
plt.title('(B)')
plt.show()
```

![ar22](media/ar22.png)

图3.4 各种AR（2）过程的模拟，$(A)\phi_1=0.5,\phi_2=0.3,x_0=x_1=0$,$(B)\phi_1=1,\phi_2=-0.5,x_0=x_1=0$，AR，自相关。



3.19由于所有AR过程都具有“减弱”的ACF，因此有时难以区分不同阶的过程。为了协助这种区分，可以使用偏自相关函数（PACF）。一般来说，两个随机变量之间的相关性通常归因于两个变量都与第三个变量相关。在目前的情况下，$x_t$和$x_{t-k}$之间的相关性可能是由于这一对具有介于中间的滞后项$x_{t-1},x_{t-2},...,x_{t-k+1}$。为了在“内部”相关中进行调整，需要计算偏自相关。



3.20在AR（k）的过程，第k个偏自相关是系数$\phi_{kk}$：

$x_y=\phi_{k1}x_{t-1}+\phi_{k2}x_{t-2}+...+\phi_{kk}x_{t-k}+a_t$(3.8)

已经调整介于中间的滞后项后,测量$x_t$和$x_{t-k}$之间的附加相关性。

一般而言，$\phi_{kk}$可以从（3.8）对应的的Yule-Walker方程获得。这些由方程（3.7），$p=k,\phi_i=\phi_{ii}$所示的集合给出，用克莱姆法则导出最后系数$\phi_{kk}$的解：

$\phi_{kk} =\frac{ \left |\begin{array}{ccc} 1 & \rho_1 & \ldots& \rho_{k-2}& \rho_1   \\  \rho_1  & 1 & \ldots& \rho_{k-3} & \rho_2   \\ \vdots &\vdots & \dots & \vdots &\vdots & \\ \rho_{k-1}& \rho_{k-2} & \ldots& \rho_1 & 1  \end{array} \right| }{\left |\begin{array}{ccc} 1 & \rho_1 & \ldots& \rho_{k-2}& \rho_{k-1}   \\  \rho_1  & 1 & \ldots& \rho_{k-3} & \rho_{k-2}   \\ \vdots &\vdots & \dots & \vdots &\vdots & \\ \rho_{k-1}& \rho_{k-2} & \ldots& \rho_1 & 1  \end{array} \right| }$



因此，对于k=1，$\phi_{11}=\rho_1=\phi$，而对于k=2，

$\phi_{22}=\frac{\left|\begin{array}{c}1&\rho_1\\\rho_1&\rho_2\end{array} \right |}{\left|\begin{array}{c}1&\rho_1\\\rho_1&1\end{array} \right |}=\frac{\rho_2-\rho_1^2}{1-\rho_1^2}$

由$\phi_{kk}$的定义，认为AR过程的PACFs遵从模式：

$\left. \begin{array}{c} AR(1):\phi_{11}=\rho_1=\phi && \phi_{kk}=0,k>1 \\AR(2):\phi_{11}=\rho_1,\phi_{22}=\frac{\rho_2-\rho_1^2}{1-\rho_1^2}  &&\phi_{kk}=0,k>2 \\ \vdots \\AR(p):\phi_{11}\ne0,\phi_{22}\ne0,...,\phi_{pp}\ne0 &&\phi_{kk}=0,k>p \end{array} \right.$

因此，大于过程的阶的滞后的偏自相关为零。因此，AR（p）过程可以通过以下方式描述：

1. 范围无限的ACF，是衰减指数和衰减正弦波的组合。

2. 滞后项大于p的PACF为零。



3.21 q阶的一般MA（MA（q））可以写成：

$x_t=a_t-\theta_1a_{t-1}-...-\theta_qa_{t-q}$

或

$x_t=(1-\theta_1B-...--\theta_qB^q)a_t=\theta (B)a_t$

ACF可以写为：

$\rho_k=\frac{-\theta_k+\theta_1\theta_{k+1}+...+\theta_{q-k}\theta_q}{1+\theta_1^2+...+\theta_q^2},k=1,2,...,q$

$\rho_k=0,k>q$

因此，MA（q）过程的ACF在滞后q之后截止；该过程的记忆扩展了q个周期，与间隔超过q个周期的观测值不相关。



3.22在$AR(\infty)$表示$\pi(B)x_t=a_t$的权重由$\pi(B)=\theta^{-1}(B)$给出，且可通过在$\pi(B)\theta(B)=1$中等同$B^j$的系数来获得。对于可逆性，

$(1-\theta_1B-...--\theta_qB^q)=(1-h_1B)...(1-h_qB)=0$

的根必须满足$|h_i|<1,i=1,2,...,q$。



3.23 图3.5显示了从两个MA（2）过程生成的序列，$a_t\sim NID(0,25)$。该序列往往参差不齐，类似于具有相反符号实根的AR（2）过程，当然，这样的MA过程无法捕获周期性行为。



3.24可以证明MA（q）过程的PACF在范围上是无限的，这样它就变弱了。MA过程的PACF的显式表达式是复杂的，但通常以指数衰减组合为主（对θ(B)上的实根）和/或阻尼正弦波（对于复根）。因此，它们的模式类似于AR过程的ACF。

```python
# MA（2）过程的模拟
a = [i for i in range(0,101)]

def ACFs(a1,a2,ran):
    x = [ran[0],ran[0]-a1*ran[1]]
    for i in range(99):
        n = ran[i+2] - a1*ran[i+1] - a2*ran[i]
        x.append(n)
    return x

ran = []
for i in range(101):
        ran.append(random.gauss(0, 25))
x1 = ACFs(-0.5,0.3,ran)
x2 = ACFs(0.5,0.3,ran)
#设置画布
plt.figure(figsize=(15,5))
# (A)    
plt.subplot(121)
plt.xlabel("t")
plt.ylabel("x_t")
plt.plot(a,x1)
plt.title('(A)')
# (B)
plt.subplot(122)
plt.xlabel("t")
plt.ylabel("x_t")
plt.plot(a,x2)
plt.title('(B)')
plt.show()
```

![ma2](media/ma2.png)

图3.5 MA（2）过程的模拟。$(A)\theta_1=-0.5,\theta_2=0.3$,$(B)\theta_1=0.5,\theta_2=0.3$。MA，滑动平均。

实际上，AR和MA过程之间存在着重要的对偶性：AR（p）过程的ACF在范围上是无限的，PACF在滞后项p后截止。另一方面，MA（q）过程的ACF会在滞后q之后截止，而PACF的范围是无限的。



## **自回归滑动平均模型**

3.25我们也可能会接受自回归模型和滑动平均模型的组合。例如，考虑AR（1）和MA（1）模型的自然组合，称为一阶自回归滑动平均，或ARMA（1,1）过程：

$x_t-\phi x_{t-1}=a_t-\theta a_{t-1}$(3.9)

或

$(1-\phi B)x_t=(1-\theta B)a_t$

$MA(\infty)$表示中的ψ权重由下式给出：

$\psi(B)=\frac{1-\theta B}{1-\phi B}$

因此

$x_t=\psi(B)a_t=(\sum_{i=0}^\infty\phi^iB^i)(1-\theta B)a_t=a_t+(\phi-\theta)\sum_{i=1}^\infty\phi^{i-1}a_{t-i}$(3.10)

同样，$AR(\infty)$表示中的π权重由下式给出：

$\pi(B)=\frac{1-\phi B}{1-\theta B}$

因此

$\pi(B)x_t=(\sum_{i=0}^\infty\phi^iB^i)(1-\phi B)x_t=a_t$

或者

$x_t=(\phi-\theta)\sum_{i=0}^\infty\theta^{i-1}x_{t-i}+a_t$

因此，ARMA（1,1）过程导致MA和自回归表示都具有无限数量的权重。该ψ权重对于$|\phi|<1$收敛（平稳条件）和π权重对于$|\theta|<1$收敛（可逆条件）。因此ARMA（1,1）过程的平稳条件与AR（1）过程的相同。



3.26从公式(3.10)得出任何乘积$x_{t-k}a_{t-j}$的期望值为零,如果k\>j。因此，将(3.9)的两边乘以$x_{t-k}$并取期望值得到：

$\gamma_k=\phi\gamma_{k-1},k>1$

而对于k=0和k=1我们分别获得

$\gamma_0-\phi\gamma_1=\sigma^2-\theta(\phi-\theta)\sigma^2$

和

$\gamma_1-\phi\gamma_0=-\theta\sigma^2$

从这两个方程中消除$\sigma^2$允许ARMA（1,1）过程的ACF经过一些代数运算后给出：

$\rho_1=\frac{(1-\phi \theta)(\phi-\theta)}{1+\theta^2-2\phi \theta}$

和

$\rho_k=\phi\rho_{k-1},k>1$

因此，ARMA（1,1）过程的ACF与AR（1）过程相似，自相关以$\phi$指数衰减相似。不同于AR（1），然而，从这种衰减从$\rho_1$开始，而不是从$\rho_0=1$。此外，并且如果$\phi$和θ都为正与$\phi>\theta$，$\rho_1$可能比$\phi$小得多,如果$\phi-\theta$较小。



3.27通过组合AR（p）和MA（q）过程，可以得到更普遍的ARMA模型：

$x_t-\phi_1x_{t-1}-...-\phi_px_{t-p}=a_t-\theta_1a_{t-1}-...-\theta_qa_{t-q}$

或

$(1-\phi_1 B-...-\phi_pB^p)x_t=(1-\theta_1B-...-\theta_qB^q)a_t$(3.11)

可以更简洁地写为

$\phi(B)x_t=\theta(B)a_t$

生成的ARMA（p，q）过程具有平稳性和可逆性，条件分别与AR（p）和MA（q）过程相关。其ACF最终将与q-P个初始值$\rho_1,...,\rho_{q-p}$之后的AR（p）过程模式的相同。而其PACF（对于k\>q-p）的行为最终类似于MA（q）过程。



3.28在整个发展过程中，一直假设该过程的均值μ为零。非零均值很容易在(3.11)通过$x_t-\mu$替换$x_t$获得，因此在ARMA（p，q）过程的一般情况下，我们有：

$\phi(B)(x_t-\mu)=\theta(B)a_t$

注意到$\phi(B)\mu=(1-\phi_1-...-\phi_p)\mu=\phi(1)\mu$，该模型可以等价写成：

$\phi(B)x_t=\theta_0+\theta(B)a_t$

其中$\theta_0=\phi(1)$是一个常数或截距。



## **ARMA模型的建立和估计**

3.29将ARMA模型拟合到观测到的时间序列，必不可少的第一步是获得通常是未知参数$\mu,\sigma_x^2,\rho_k$的估计。由于平稳性和（隐含的）遍历性的假设，$\mu,\sigma_x^2$可以分别由的样本均值和样本方差估算。$\rho_k$的估计值由k滞后的样本自相关提供。由于其重要性，在这里写出：

$r_k=\frac{\sum_{t=k+1}^T(x_t-\bar x)(x_{t-k}-\bar x)}{Ts^2},k=1,2,...$

回忆1.2，$r_k$们的集合定义了样本自相关函数（SACF），有时称为相关图。



3.30考虑一个时间序列，为有限方差（即，$\rho_k=0,k\ne0$）的固定分布生成的独立观测。这样一个序列被认为是独立同等分布的，记为i.i.d.。这样的序列，$r_k$的方差大约由$T^{-1}$给出。如果T也很大，$\sqrt Tr_k$将近似为标准正态分布，因此$r_k {}_\sim^a N(0,T^{-1})$，表示$r_k$的绝对值超过$2/\sqrt T$可以被认为在5％显著性水平下,“显著”不为0。更一般，对于k\>q如果$\rho_k=0$，$r_k$的方差是：

$V(r_k)=T^{-1}(1+2\rho_1^2+...+2\rho_q^2)$

因此，通过连续增加q的值和按其样本估计更换$\rho_k$们，序列$r_1,r_2,...,r_k$的方差可以估计为$T^{-1},T^{-1}(1+2r_1^2),...,T^{-1}(1+2r_1^2+...+2r_{k-1}^2)$，当然，这些对于k\>1，比那些使用简单的公式$T^{-1}$计算将更大。取$V(r_k)$的平方根给出附加到$r_k$上的标准误，这些通常称为Bartlett标准误。



3.31样本偏自相关函数（SPACF）通常是通过拟合阶数递增的自回归模型计算；每个模型的最后一个系数的估计是样品偏自相关，$\hat\phi_{kk}$。如果数据服从AR（p）过程，则滞后大于p的$\hat\phi_{kk}$的方差大约为$T^{-1}$，因此$\hat\phi_{kk}{}_\sim^a N(0,T^{-1})$。



3.32 给出$r_k$和$\hat\phi_{kk}$，可以用Box和Jenkins方法建立ARMA模型。这是一个三阶段程序，第一个是所谓的识别阶段，本质上是为了将时间序列的SACF和SPACF的行为与各种理论的ACF和PACF相匹配。这可以通过评估各个样品的自相关和偏自相关的显著性，通过将它们与其伴随的标准误差作比较。此外，可以基于完整的“
portmanteau”统计构造的集合。基于假定$x_t{}_\sim^a WN(\mu,\sigma^2)$，Ljung和Box证明：

$Q(k)=T(T+2)\sum_{i=1}^k(T-i)^{-1}r_i^2{}_\sim^a\chi^2(k)$

并且该统计数据可用于评估观测到的序列是否显著偏离白噪声。



3.33选择了最佳匹配（或一组匹配）后，第二阶段是估计模型未知参数。如果模型是纯自回归，则普通最小二乘（OLS）是有效可行的估计方法，因为它可以产生参数的条件ML估计; 这里的“有条件”是指在确定$x_1,x_2,...,x_p$的条件下最大化似然函数，因此，由其观测值给出，而不是取自潜在分布的随机变量。如果样本量T很大，则前p个观察值对总的似然的贡献可忽略不计。

如果存在MA成分，则一种简单的方法是假设$a_{p-q+1},a_{p-q+2},...a_p$期望都为零。这称为条件最小二乘（CLS），在大样本中与精确的ML等效。



3.34最后，第三阶段是诊断检查，检查残差：

$\hat a_t=x_t-\hat \phi_1x_{t-1}-...-\hat \phi_px_{t-p}-\hat\theta_1\hat a_{t-1}-...-\hat\theta_q\hat a_{t-q}$

从拟合模型中可能因任何可能的设定误差而得到。设定误差通常采用自相关残差的形式，因此$\hat a_t$的SACF将包含一个或多个显著值。显著性可以通过将单个残差自相关（比如$\hat r_k$）与它们的标准误进行比较，在零假设残差不是设定误差下将是$T^{-1/2}$，因此是白噪声。另外，可以计算出Portmanteau统计量，尽管现在必须减小Q的自由度到k-p-q，如果已拟合了ARMA（p，q）。

进一步检查拟合模型是否过度拟合的充分性，例如估计ARMA（p+1，q）或ARMA（p，q+1）过程并检查拟合参数是否显著。如果遇到模型中的任何缺陷，则必须进行进一步的模型构建，直到获得没有明显缺陷的规格明确的模型。



3.35这种三阶段方法，于1960年代开发，那时计算功能极为有限，并且没有可用的软件（实际上，Box和Jenkins不得不从头开始编写所有程序），可能会引起现代读者认为这是不必要的劳动密集，尽管很明显它确实具有一些重要的优势，因为它使分析师能够获得数据的详细“感觉”。因此，另一种方法可以利用现代计算机的功能和可用的软件来选择，基于先前对p和p的最大设置的考虑的一组模型，估计每个可能的模型，然后基于拟合优度的选择标准选择最小化的模型。



3.36有多种选择标准可用于选择合适的模型，最受欢迎的模型也许是Akaike（1974）的信息准则（AIC），定义为：

$AIC(p,q)=log\hat\sigma^2+2(p+q)T^{-1}$

尽管具有更好的理论属性的标准是施瓦兹（1978）的BIC

$BIC(p,q)=log\hat\sigma^2+(p+q)T^{-1}logT$

这些标准按以下方式使用。上确界，即$p_{max}$和$q_{max}$，是$\phi(B)$和$\theta(B)$的阶数的集合，$\bar p=\{0,1,...,p_{max}\}$，$\bar q=\{0,1,...,q_{max}\}$,例如选择阶数$p_1,q_1$：

$AIC(p_1,q_1)=minAIC(p,q),p\in\bar p,q\in\bar q$

与BIC结合使用的并行策略。应用此策略的一个可能困难是没有具体的有关于如何确定$\bar p,\bar q$的指导原则，尽管它们被默认假定足够大，以包含的“真实”模型范围，即可以包含阶数$(p_0,q_0)$，这些当然，不一定与在考虑的标准下选择的阶数$(p_1,q_1)$相同。

鉴于这些替代标准，是否有理由偏向一个？如果真阶$(p_0,q_0)$包含在集合$(p,q),p\in\bar p,q\in\bar q$中；那么对于所有标准，$p_1\ge p_0,q_1\ge q_0$，几乎可以确定，当$T\to \infty$。然而，BIC在渐进确定真实模型方面具有很强的一致性，而对于AIC，将出现一个过参数化的模型，无论实现有多长时间。当然，这些属性不一定要在有限的样本中得到保证，因此这两个条件经常一起使用。
