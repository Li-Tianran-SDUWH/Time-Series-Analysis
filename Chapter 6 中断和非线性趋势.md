# **Chapter 6 中断趋势**

## **中断趋势模型**

6.1趋势平稳（TS）对差分平稳（DS）二分法，5.9‑5.10中概述的相关检验程序既简单又容易实施，但是一定现实吗？TS在某些情况下，“全局”线性趋势的备择假设过于简单，从而表明可能需要更复杂的趋势函数？通常，趋势的更合理的候选者是在一个或多个时间点“中断”的线性函数。

趋势可能会以几种方式中断。简单假设，例如，在已知时间点$T_b^c(1<T_b^c<T)$有一个中断，上标“c”表示“正确的”中断日期，区别在于在适当的时候将变得重要。最简单的中断趋势模型是“水平移动”，其中$x_t$的水平在$T_b^c$从$\mu_0$至$\mu_1=\mu_0+\mu$。可以将其参数化为

$x_t=\mu_0+(\mu_1-\mu_0)DU_t^c+\beta_0t+\varepsilon_t=\mu_0+\mu DU_t^c+\beta_0t+\varepsilon_t$(6.1)

其中如果$t\le T_b^c$，则$DU_t^c=0$；如果$t> T_b^c$，则$DU_t^c= 1$。该移位变量可简写为$DU_t^c=\mathbf1(t> T_b^c)$，其中$\mathbf 1$是指标函数，参数为true时取值为1，否则为0。另一种可能性是“变化增长”模型，其中趋势变化的斜率在$T_b^c$从$\beta_0$到$\beta_1=\beta_0+\beta$水平不变。在这种情况下，趋势函数在中断时被加入，通常被称为细分趋势。该模型可以参数化为

$x_t=\mu_0+(\beta_1-\beta_0)DT_t^c+\beta_0t+\varepsilon_t=\mu_0+\beta DT_t^c+\beta_0t+\varepsilon_t$(6.2)

其中$DT_t^c=\mathbf1(t> T_t^c)(t-T_t^c)$模拟增长的变化。当然两种形式中断可能同时发生，这样我们就可以组合模型

$x_t=\mu_0+(\mu_1-\mu_0)DU_t^c+\beta_0t+(\beta_1-\beta_0)DT_t^c+\varepsilon_t=\mu_0+\mu DU_t^c+\beta_0t+\beta DT_t^c+\varepsilon_t$(6.3)

因此$x_t$在$T_b^c$处同时发生水平和斜率的变化。



6.2在模型（6.1）-（6.3）误差过程$\varepsilon_t$已保留为未指定。一个显而易见的选择是，它是一个ARMA过程，$\phi(B)\varepsilon_t=\theta(B)a_t$，其中(6.1)-(6.3）为中断趋势平稳模型。假设自回归多项式可以被因式分解为$\phi(B)=(1-\phi B)\phi_1(B)$。如果$\phi(B)$包含一个单位根，那么$\phi=1,\phi(B)=\triangledown\phi_1(B)$（参见4.12），（6.1）成为

$\triangledown x_t=\beta_0+\mu\triangledown DU_t^c+\varepsilon_t^*=\beta_0+\mu D(TB^c)_t+\varepsilon_t^*$(6.4)

其中$\phi_1(B)\varepsilon_t^*=\theta(B)a_t$和我们已经定义$D(TB^c)_t=\triangledown DU_t^c=\mathbf1(t= T_b^c+1)$。因此，该模型将指定$x_t$为一个$I(1)$过程，具有漂移且在$T_b^c$取值为1，其他取值0的虚拟变量。同样，(6.2)误差过程中的单位根导出

$\triangledown x_t=\beta_0+\beta\triangledown DT_t^c+\varepsilon_t^*=\beta_0+\beta DU^c_t+\varepsilon_t^*$(6.5)

使得从所述漂移变化在中断点$T_b^c$从$\beta_0$到$\beta_1$。混合模型变成

$\triangledown x_t=\beta_0+\mu D(TB^c)_t+\beta DU^c_t+\varepsilon_t^*$(6.6)



## **中断趋势和单位根检验**

6.3我们如何区分TS中断趋势和中断DS过程？显然，单位根检验应该适用，但是此类检验对中断趋势有什么影响？Perron率先考虑中断趋势的影响和单位根检验的变换水平，表明该类型的标准检验，以及分数差分法与TS备择假设不一致时，包含斜率的趋势函数变化。在这里，最大自回归根偏向于1，实际上，单位根零假设变为不可拒绝，甚至是渐进的。尽管检验是一致的，针对趋势函数的截距的变化，它们的效力仍然大大减少了，因为自回归根的估计的极限值远大于其真实值。



6.4 Perron（1989）因此扩展了Dickey‑Fuller单位根检验法，建立两个渐近等效的过程，确保一致反对变化趋势函数。首先使用初始回归去$x_t$趋势，根据任一模型（A）,水平变化(6.1);模型（B），细分趋势（6.2）;或模型（C），混合模型(6.3)。因此，让，$\widetilde x_t^i$,i=A，B，C为$x_t$回归的残差在（1）i=A上：常数t（2）i=B：常数t和$DT_t^c$（3）i=C：常数t，$DT_t^c$和$DU_t^c$。对于模型（A）和（C），估计改进的ADF回归：

$\widetilde x_t^i=\widetilde\phi^i\widetilde x_{t-1}^i+\sum_{j=0}^k\gamma_jD(TB^c)_{t-j}\sum_{j=1}^k\delta_j\triangledown\widetilde x_{t-j}^i,i=A,B,C$

和$\widetilde\phi^i=1$的t检验（$t^i,i=A,C$）。包含k+1个虚拟变量$D(TB^c)_t,...,D(TB^c)_{t-k}$来确保$t^A,t^C$的极限分布相对于误差自相关结构不变（参见Perron和Vogelsang，1993）。对于模型（B），“未改进” 的ADF回归

$\widetilde x_t^B=\widetilde\phi^i\widetilde x_{t-1}^B+\sum_{j=1}^k\delta_j\triangledown\widetilde x_{t-j}^B+a_t$

可以估计得到$t_B$。



6.5  $t^i,i=A,C$的渐近临界值由Perron提供，$t^B$的渐近临界值由Perron 和Vogelsang提供。这些取决于中断发生的位置，因此是中断分数$\tau^c=T_b^c/T$的函数。例如，对于模型（A）和$\tau^c=0.5$（一中断发生在样品的中点），检验$\widetilde\phi^A=1$的5％，2.5％和1％的临界值分别是-3.76、-4.01和-4.32，如果在开始附近中断（$\tau^c=0.1$）或尾部（$\tau^c=0.9$），则这些临界值的绝对值较小，分别为 -3.68 、-3.93 、-4.30 和 -3.69、-3.97和-4.27。可以预期的，当$\tau^c=0或1$与标准DF临界值相同，因为在这些极端情况下没有中断发生。模型（B）和（C）的临界值绝对值较大：对于后一种模型，用于检验$t^C$的中点临界值是-4.24、-4.53和-4.90。自然，对于给定的检验尺寸，所有统计量的绝对值要比标准DF临界值更高。



6.6 Perron（1989）指出了先前趋势分离方法的可能不利之处，这意味着趋势函数的变化瞬间发生，所以这种变化类似于时间序列的“附加异常值”（AO）影响。因此可以考虑转化期，序列在其中逐渐对趋势函数的冲击反应。将模型（A）作为例子，可以指定为：

$x_t=\mu_0+\mu\psi(B)DU_t^c+\beta_0t+\varepsilon_t$

其中$\psi(B)$是平稳的并且可逆的,$\psi(0)=1$。冲击的直接影响为μ，但长期变化为$\mu\psi(1)$。



6.7将这种逐渐变化纳入趋势函数的一种方法是，假设$x_t$对趋势冲击的反应方式与对任何其他冲击的反应方式相同。回顾6.2的ARMA对$\varepsilon_t$的说明，即，$\phi(B)\varepsilon_t=\theta(B)a_t$，这将意味着$\psi(B)=\phi(B)^{-1}\theta(B)$，这将类似于“新息异常值”（IO）模型。使用此规范检验单位根存在情况，可以使用ADF回归框架的直接扩展来合并虚拟变量：

$x_t=\mu^A+\theta^ADU_t^c+\beta^At+d^AD(TB^c)_t+\phi^Ax_{t-1}+\sum_{i=1}^k\delta_i\triangledown x_{t-i}+a_t$ (6.9)

$x_t=\mu^B+\theta^BDU_t^c+\beta^Bt+\gamma^BDT^c_t+\phi^Bx_{t-1}+\sum_{i=1}^k\delta_i\triangledown x_{t-i}+a_t$ (6.10)

$x_t=\mu^C+\theta^CDU_t^c+\beta^Ct+\gamma^CDT^c_t+d^CD(TB^c)_t+\phi^Cx_{t-1}+\sum_{i=1}^k\delta_i\triangledown x_{t-i}+a_t$ (6.11)

单位根的零假设在每个模型限制以下参数：模型（A）：$\phi^A=1,\theta^A=\beta^A=0$; 模型（B）：$\phi^B=1,\gamma^B=\beta^B=0$; 和模型（C）：$\phi^C=1,\gamma^C=\beta^C=0$。用于检验$\phi^A=1,\phi^C=1$的渐近分布与(6.7)的$\widetilde\phi^A=1,\widetilde\phi^C=1$相同，但该对应关系不适用于的(6.10)的t统计量。确实，Perron认为检验细分趋势模型应该只使用（6.8）。



6.8 Perron（1989）认为，拒绝单位根零假设，在潜在趋势函数在已知日期发生变化的条件下，并不意味着该序列一定可以建模为完全确定趋势函数的平稳波动。为此，Perron引用了一般的统计原理，即否定原假设并不意味着接受任何特定的备择假设。Perron想到的是保持的假设类别可以参数化为

$x_t=\eta_t+\varepsilon_t,\eta_t=\mu_t+\beta_tt,\triangledown\mu_t=\upsilon(B)v_t,\triangledown\beta_t=\omega(B)w_t$(6.12)

其中$\phi(B)\varepsilon_t=\theta(B)a_t$。趋势函数的截距和斜率，$\mu_t,\beta_t$，取单积过程，其中$\upsilon(B),\omega(B)$是平稳可逆的。但是，冲击的时间和发生时刻$v_t,w_t$假定与新息序列零星相关，可能是到达率指定的泊松过程，发生的频率与$a_t$序列实现的频率几乎不相关。



6.9模型（6.12）背后的直觉想法是趋势函数的系数由长期的“基本原理”决定，很少更改。关于趋势函数变化的外生假设是允许我们从噪音中消除这些罕见的冲击，进入不必专门为$\mu_t,\beta_t$的随机行为建模的趋势的策略。Perron的框架是检验噪声$\varepsilon_t$是否是一个单积过程，通过删除$v_t,w_t$的非零值被认为已经发生的日期的事件，并作为趋势函数的一部分对其建模。