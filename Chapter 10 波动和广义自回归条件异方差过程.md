# **Chapter 10 波动和广义自回归条件异方差过程**

## **波动**

10.1 随着1950年代对组合理论进行了初步研究，波动成为金融中极为重要的概念，经常出现在例如资产定价和风险管理模型中。虽然波动有各种定义，在时间序列的背景下，通常被认为是在序列的演变中具有高可变性的一个时期，或等效地，具有高方差。这是由观测许多时间序列得出，而不仅仅是金融收  益，其特点是交替的波动性较低的相对平静时期和波动性较高的相对波动时期。



10.2 最初对波动有关的兴趣不是直接观察到的，因此采取了几种替代措施以凭经验近似。在1980年代初期，有人提议波动应该嵌入到正式的观测时间序列的随机模型中。这是基于事实，尽管有些序列似乎序列不相关，但它们不随时间独立。因此，他们有潜力在更高的矩展现出丰富的动力，这些时刻通常伴随着有趣的非高斯分布特性。在这种情况下，应该专注于序列更高的矩的特征，而不是仅对条件均值建模。



10.3一种简单的方法是允许过程的方差（或典型的条件方差）产生序列$x_t$来持续或在某些离散时间点进行改变。虽然一个平稳过程必须具有恒定的方差，某些条件方差可以改变，因此尽管无条件方差$V(x_t)$可能对于所有t都是常数，取决于$x_t$的实现的条件方差$V(x_t|x_{t-1},x_{t-2},...)$，能够随观测变化。



10.4具有随时间变化的条件方差的随机模型可以通过假设$x_t$由乘积过程生成来定义：

$x_t=\mu+\sigma_tU_t$(10.1)

其中$U_t$是标准过程，因此对于所有的t,$E(U_t)=0,V(U_t)=E(U_t^2)=1$，并且$\sigma_t$是正随机变量使：

$V(x_t|\sigma_t)=E((x_t-\mu)^2|\sigma_t)=\sigma_t^2E(U_t^2)=\sigma_t^2$

因此$\sigma_t^2$是$x_t$的条件方差，$\sigma_t$是$x_t$的条件标准差。

一般地，$U_t=(x_t-\mu)/\sigma_t$被假定为正态的且与$\sigma_t$独立：我们将进一步假定它是严格的白噪声，从而$E(U_tU_{t-k})=0,k\ne0$。这些假设意味着$x_t$具有均值μ，方差：

$E(x_t-\mu)^2=E(\sigma_t^2U_t^2)=E(\sigma_t^2)E(U_t^2)=E(\sigma_t^2)$

和自协方差

$E(x_t-\mu)(x_{t-k}-\mu)=E(\sigma_t\sigma_{t-k}U_tU_{t-k})=E(\sigma_t\sigma_{t-k})E(U_tU_{t-k})=0$

因此为白噪声。但是，注意，平方和绝对偏差，$S_t=(x_t-\mu)^2$和$M_t=|x_t-\mu|$，可能是自相关的。例如

$Cov(S_t,S_{t-k})=E(S_t-E(S_t))(S_{t-k}-E(S_t))=E(S_tS_{t-k})-(E(S_t))^2=E(\sigma_t^2\sigma_{t-k}^2)E(U_t^2U_{t-k}^2)-(E(\sigma_t^2))^2=E(\sigma_t^2\sigma_{t-k}^2)-(E(\sigma_t^2))^2$

因此

$E(S_t^2)=E(\sigma_t^4)-(E(\sigma_t^2))^2$

$S_t$的第k个自相关为

$\rho_{k,S}=\frac{E(\sigma_t^2\sigma_{t-k}^2)-(E(\sigma_t^2))^2}{E(\sigma_t^4)-(E(\sigma_t^2))^2}$

这种自相关只会是零，如果$\sigma_t^2$是常数，在这种情况下$x_t$可以被写为$x_t=\mu+a_t$，其中$a_t=\sigma U_t$具有零均值和常数方差，这是一个定义$a_t$为白噪声另一种方式。



## **自回归条件异方差过程**

10.5到现在为止，我们没有讨论条件方差$\sigma_t^2$怎样可能会生成。我们现在考虑它们是$x_t$过去值的函数的情况：

$\sigma_t^2=f(x_{t-1,}x_{t-2},...)$

一个简单的例子是：

$\sigma_t^2=f(x_{t-1})=\alpha_0+\alpha_1(x_{t-1}-\mu)^2$(10.2)

其中$\alpha_0,\alpha_1$均为正，确保$\sigma_t^2>0$。$U_t\sim NID(0,1)$且与$\sigma_t$独立，则$x_t=\mu+\sigma_tU_t$为条件正态，

$x_t|x_{t-1},x_{t-2},...\sim NID(\mu,\sigma_t^2)$

因此

$V(x_t|x_{t-1})=\alpha_0+\alpha_1(x_{t-1}-\mu)^2$

如果$0<\alpha_1<1$，则无条件方差为$V(x_t)=\alpha_0/(1-\alpha_1)$且$x_t$是弱平稳的。可以证明如果$3\alpha_1^2<1$,则$x_t$的四阶矩是有限的，如果是这样，$x_t$的峰度为$3(1-\alpha_1^2)/(1-3\alpha_1^2)$。由于该值必须超过3，$x_t$的无条件分布会比正态更厚尾。如果此矩的条件不满足，则$x_t$的方差将无穷大，并且$x_t$不会是弱平稳的。



10.6此模型称为一阶自回归条件异方差[ARCH(1)]过程，最初由Engle（1982，1983）引入。ARCH过程已被证明在时间序列中模拟波动非常流行。一个更方便的记法是定义$\varepsilon_t=x_t-\mu=U_t\sigma_t$，使得ARCH（1）模型可以写成：

$\varepsilon_t|x_{t-1},x_{t-2},...\sim NID(0,\sigma_t^2)$

$\sigma_t^2=\alpha_0+\alpha_1\varepsilon_{t-1}^2$

定义$\upsilon_t=\varepsilon_t^2-\sigma_t^2$，模型也可以写成：

$\varepsilon_t^2=\alpha_0+\alpha_1\varepsilon_{t-1}^2+\upsilon_t$

既然$E(\upsilon_t|x_{t-1},x_{t-2},...)=0$，模型直接对应于一个AR（1）过程，对于平方新息$\varepsilon_t^2。$然而，$\upsilon_t=\sigma_T^2(U_t^2-1)$，偏差显然具有随时间变化的方差。



10.7一个自然扩展是ARCH（q）过程，其中（10.2）被下式代替：

$\sigma_t^2=f(x_{t-1},x_{t-2},...x_{t-q})=\alpha_0+\sum_{i=1}^q\alpha_i(x_{t-i}-\mu)^2$

其中$\alpha_i\ge0,0\le i\le q$。如果所有与ARCH参数相关的特征方程的根比单位一小，这个过程将是弱平稳的。这意味着$\sum_{i=1}^q\alpha_i<1$，在这种情况下，无条件方差为$V(x_t=\alpha_0/(1-\sum_{i=1}^q\alpha_i)$。就$\varepsilon_t$和$\sigma_t^2$来讲，条件方差函数是

$\sigma_t^2=\alpha_0+\sum_{i=1}^q\alpha_i\varepsilon_{t-i}^2$

或者，等价地，定义$\alpha(B)=\alpha_1+\alpha_2B+...+\alpha_qB^{q-1}$，



10.8 ARCH模型的实际困难是,q大时，无约束的估计通常会导致违反$\alpha_i$的非负性约束,确保条件方差$\sigma_t^2$始终为正。在模型的早期应用中，任意下降滞后结构被强加到$\alpha_i$确保这些约束满足。为了获得更大的灵活性，可以进一步扩展为广义ARCH（GARCH）过程，由Bollerslev（1986）引入。GARCH（p，q）过程具有条件方差函数：

$\sigma_t^2=\alpha_0+\sum_{i=1}^q\alpha_i\varepsilon_{t-i}^2+\sum_{i=1}^p\beta_i\sigma_{t-i}^2=\alpha_0+\alpha(B)\varepsilon_{t-1}^2+\beta(B)\sigma_{t-1}^2$

其中$p>0,\beta_i\ge0,i\le1\le p$。为了GARCH（p，q）过程的条件方差很好地定义，相应的$ARCH(\infty)$模型$\sigma_t^2=\theta_0+\theta(B)\varepsilon_t^2$的所有系数必须为正。假如$\alpha(B),\beta(B)$没有共同的根且$1-\beta(B)$的根都小于1，当且仅当所有$\theta(B)=\alpha(B)/(1-\beta(B))$的系数非负，正约束才满足。对于GARCH（1,1）过程，

$\sigma_t^2=\alpha_0+\alpha_1\varepsilon_{t-1}^2+\beta_1\sigma_{t-1}^2$

在描述金融时间序列非常流行的模型，这些条件仅要求所有三个参数为非负的。

GARCH（p，q）过程的等效形式为

$\varepsilon_t^2=\alpha_0+(\alpha(B)+\beta(B))\varepsilon_{t-1}^2+\upsilon_t-\beta(B)\upsilon_{t-1}$(10.3)

使得$\varepsilon_t^2$是ARMA（m，p），其中m=max（p，q）。这个过程弱平稳，当且仅当，$1-\alpha(B)-\beta(B)$的根都小于1，因此$\alpha(1)+\beta(1)<1$。



10.9如果在(10.3)中$\alpha(1)+\beta(1)=1$,则$1-\alpha(B)-\beta(B)$将包含一个单位根，我们说模型是单积的GARCH或IGARCH。通常的情况是$\alpha(1)+\beta(1)$接近一，如果是这样，一个对条件方差的冲击是持久的，从这种意义上说对所有未来观测它仍然很重要。



10.10虽然我们假设$\varepsilon_t$的分布是条件正态，这不是必需的。例如，分布可以是 Student’s t，未知自由度$v$，可以从数据中估计：对于$v>2$这样的分布是尖峰的，因此比正态厚尾。或者，误差分布可以为带有参数ς （/'sɪɡmə/）的广义指数（GED），可以再次根据数据估计。正态分布的特征是ς=2，ς\<2说明分布是厚尾的。无论假定的误差分布如何，估计将需要非线性迭代技术和最大似然估计。



10.11分析还基于进一步假设$\varepsilon_t=x_t-\mu_t$序列不相关，一个自然延伸是允许$x_t$遵循ARMA（P，Q）过程，因此组合的ARMA（P，Q）-ARCH（p，q）模型变为：

$\Phi(B)(x_t-\mu)=\Theta(B)\varepsilon_t$

$\sigma_t^2=\alpha_0+\alpha(B)\varepsilon_{t-1}^2+\beta(B)\sigma_{t-1}^2$



## **检验是否存在ARCH误差**

10.12让我们假设已经估计了一个$x_t$的ARMA模型，残差$e_t$已经获得。如果忽略ARCH的存在可能导致严重的模型设定错误。与所有形式的异方差（即非恒定误差方差），假设不存在，则进行分析导致的不适当的参数标准误，通常太小。例如，忽略ARCH将导致识别过度参数化的ARMA模型，因为应设置为零的参数将显示为显著。



10.13因此，检验是否存在ARCH的方法必不可少，尤其是结合ARCH新息的估计需要复杂的迭代技术。等式（10.3）表明，如果$\varepsilon_t$是GARCH（P，Q），则$\varepsilon_t^2$是ARMA（m，p），其中m=max（p，q），标准ARMA理论在这种情况下，将继续遵循。这意味着纯ARMA过程估计的残差平方$e_t^2$可用于识别m和p，因此q，以类似于的方式残差本身在常规ARMA建模中使用。例如，$e_t^2$的样本自相关具有渐近方差$T^{-1}$,且计算出的Portmanteau统计量为渐近$\chi^2$，如果$\varepsilon_t^2$独立。



10.14 也可以进行正式检验。检验零假设$\varepsilon_t$有恒定的条件方差，备择假设是条件方差由ARCH（q）过程给出，即在$\beta_1=...=\beta_p=0$条件下检验$\alpha_1=...=\alpha_p$，可以基于拉格朗日乘数（LM）原理。检验程序是在$e_{t-1}^2,...,e_{t-p}^2$上进行$e_t^2$的回归，并检验统计量$T\cdot R^2$为一个$\chi_q^2$变量，其中$R^2$是回归的平方相关系数。渐近等效的检验形式，可能会有更好的小样本属性，是根据回归来计算标准F检验（这些检验由Engle于1982年提出）。该检验背后的直觉很明显。如果数据中没有ARCH效应，则方差是恒定的，$e_t^2$的变化将完全是随机的。如果存在ARCH效应，此类变化将通过平方残差的滞后值来预测。

当然，如果残差本身包含一些自相关，或非线性的其他形式，那么很可能ARCH的检验将被拒绝，因为这些错误假定可能会导致平方残差中的自相关。我们不能简单地假设当ARCH检验拒绝时，必然存在ARCH效应。



10.15当备择假设是GARCH（p，q）过程时，会有些复杂。实际上，反对白噪声零假设的对p\>0,q\>0的一般检验，是不可行的；当原假设为GARCH（p，q）时，$GARCH(p+r_1,q+r_2)$误差的检验，其中$r_1>0,r_2>0$，也不可行。此外，在此原假设下，GARCH（p，r）的LM检验和ARCH（p+r）备择假设一致。可以检验ARCH（p）过程的原假设，备择假设为GARCH（p，q）（见Bollerslev，1988）。



10.16对标准GARCH模型的一些修改是由于允许$\varepsilon_t$和$\sigma_t^2$之间的关系比迄今已假定的二次关系更灵活。为了简化说明，我们将专注于GARCH（1,1）过程的变体：

$\sigma_t^2=\alpha_0+\alpha_1\varepsilon_t^2+\beta_1\sigma_{t-1}^2=\alpha_0+\alpha_1\sigma_{t-1}^2U_{t-1}^2+\beta_1\sigma_{t-1}^2$(10.4)

早期的替代方法是对条件标准差建模而不是方差（Schwert，1989）：

$\sigma_t=\alpha_0+\alpha_1|\varepsilon_{t-1}|+\beta_1\sigma_{t-1}=\alpha_0+\alpha_1\sigma_{t-1}|U_{t-1}|+\beta_1\sigma_{t-1}$(10.5)

这使条件方差成为绝对冲击的加权平均值的平方，而不是平方冲击的加权平均值。因此，大冲击对条件方差的影响比标准GARCH模型中的要小。

不专注于方差或标准差，可以估计附加参数得到更灵活和普遍的GARCH模型（参见Ding等，1993）：

$\sigma_t^{\gamma}=\alpha_0+\alpha_1|\varepsilon_{t-1}|^{\gamma}+\beta_1\sigma_{t-1}^{\gamma}$



10.17 Nelson（1991）的指数GARCH（EGARCH）模型明确表明了冲击的非对称响应：

$log(\sigma_t^2)=\alpha_0+\alpha_1g(\frac{\varepsilon_{t-1}}{\sigma_{t-1}})+\beta_1log(\sigma_{t-1}^2)$(10.6)

其中

$g(\frac{\varepsilon_{t-1}}{\sigma_{t-1}})=\theta_1\frac{\varepsilon_{t-1}}{\sigma_{t-1}}+(|\frac{\varepsilon_{t-1}}{\sigma_{t-1}}|-E|\frac{\varepsilon_{t-1}}{\sigma_{t-1}}|)$

“信息冲击曲线”,$g(\cdot)$,与条件波动相关，这里为$log(\sigma_t^2)$,$\varepsilon_{t-1}$为“信息”。它体现了非对称的响应，因为当$\varepsilon_{t-1}>0$时$\partial g/\partial \varepsilon_{t-1}=1+\theta_1$；当$\varepsilon_{t-1}<0$时$\partial g/\partial \varepsilon_{t-1}=1-\theta_1$（注意当没有信息$\varepsilon_{t-1}=0$时，波动会最小）。这个不对称性可能会很有用，因为它可以使波动性对$x_t$下降反应更迅速，相比对应的上升，这是一个重要的程式化事实，对于许多金融资产而言，被称为杠杆效应。EGARCH模型还具有优点：不需要参数限制来确保方差为正。很容易证明$g(\frac{\varepsilon_{t-1}}{\sigma_{t-1}})$是具有零均值和恒定方差的严格白噪声，所以$log(\sigma_t^2)$是ARMA（1,1）过程，如果$\beta_1<1$将平稳。



10.18 嵌套（10.4）-（10.6）的是Higgins和Bera（1992）的非线性ARCH模型，其一般形式为：

$\sigma_t^{\gamma}=\alpha_0+\alpha_1g^{\gamma}(\varepsilon_{t-1})+\beta_1\sigma_{t-1}^{\gamma}$

而另一种选择是阈值ARCH过程

$\sigma_t^{\gamma}=\alpha_0+\alpha_1h^{(\gamma)}(\varepsilon_{t-1})+\beta_1\sigma_{t-1}^{\gamma}$

其中

$h^{(\gamma)}(\varepsilon_{t-1})=\theta_1|\varepsilon_{t-1}|^{\gamma}\mathbf 1(\varepsilon_{t-1}>0)+|\varepsilon_{t-1}|^{\gamma}\mathbf 1(\varepsilon_{t-1}\le0)$

$\mathbf 1(\cdot)$是6.1中引入的示性函数。如果$\gamma=1$，我们有Zakoian（1994）的阈值ARCH（TARCH）模型，对$\gamma=2$，我们有Glosten等人的GJR模型。（1993），它允许对信息有二次方响应的波动，但对好和坏的信息系数不同，尽管它坚持认为没有信息时产生最小波动。



10.19  GARCH（1,1）模型（10.4）的替代形式由Engle和Lee（1999）提出，定义$\alpha_0=\varpi(1-\alpha_1-\beta_1)$，其中$\varpi$是无条件方差，或长期波动，过程恢复为：

$\sigma_t^2=\varpi+\alpha_1(\varepsilon_{t-1}^2-\varpi)+\beta_1(\sigma_{t-1}^2-\varpi)$

可以扩展此形式以允许恢复到不同水平，由$q_t$定义：

$\sigma_t^2=q_t+\alpha_1(\varepsilon_{t-1}^2-q_{t-1})+\beta_1(\sigma_{t-1}^2-q_{t-1})$

$q_t=\varpi+\xi(q_{t-1}-\varpi)+\varsigma(\varepsilon_{t-1}^2-\sigma_{t-1}^2)$

这里$q_t$是波动的永久分量，收敛于ϖ，通过功率ξ，而$\sigma_t^2-q_t$是瞬时分量，收敛到零，功率为$\alpha_1+\beta_1$。该分量GARCH模型也可以与TARCH结合使用以允许永久和临时分量都有非对称性，此非对称分量GARCH模型自动将非对称性引入瞬态方程。



## **从一个ARMA-GARCH模型进行预测**

10.20假设我们有10.11的ARMA（P，Q）-GARCH（p，q）模型：

$x_t=\Phi_1x_{t-1}+...+\Phi_Px_{t-P}+\Theta_0+\varepsilon_t-\Theta_1\varepsilon_{t-1}-...-\Theta_Q\varepsilon_{t-Q}$(10.7)

$\sigma_t^2=\alpha_0+\alpha_1\varepsilon_{t-1}^2+...+\alpha_p\varepsilon_{t-p}^2+\beta_1\sigma_{t-1}^2+...+\beta_q\sigma_{t-q}^2$(10.8)

$x_{T+h}$的预测可以从7.1‑7.4中概述的“均值方程”(10.7)获得。在计算预测误差方差时，不能再假定误差方差本身不变。因此，必须将（7.4）修改为：

$V(e_{t,h})=\sigma_{T+h}^2+\psi_1^2\sigma_{T+h-1}^2+...+\psi_{h-1}^2\sigma_{T+1}^2$

$\sigma_{T+h}^2$从(10.8)递归获得。