# **Chapter 12 传递函数和自回归分布滞后模型**

## **传递函数噪声模型**

12.1 之前所有模型是单变量的，因此时间序列的当前值，线性或仅取决于自身的过去值，或许是时间的确定函数。尽管单变量模型本身很重要，它们在提供多元模型可以比较的“基准”方面也起着关键作用。我们下一章将分析几个多元模型，但我们的开发将从最简单的开始。这是单输入传递函数噪声模型，其中内生或输出变量$y_t$与单个输入或外生变量$x_t$相关，通过动态模型

$y_t=v(B)x_t+n_t$(12.1)

其中滞后多项式$v(B)=v_0+v_1B+v_2B^2+...$允许x影响y通过分布滞后：$v(B)$通常被称为传递函数，并且系数$v_i$称为冲量响应权重。



12.2假设输入变量和输出变量都是平稳的，或许经过适当的转化。但是两者之间的关系不是，确定性的，相反，它会被随机过程$n_t$捕捉的噪声污染，通常将是连续相关的。（12.1）中至关重要的假设是$x_t$和$n_t$是独立的，因此过去的x影响未来的y，但反之不亦然，因此排除了从y到x的反馈。



12.3一般而言，$v(B)$将是无穷大的阶，因此（12.1）的经验建模之前必须对传递函数施加一些限制。施加限制的典型方式类似单变量随机过程的线性滤波器表示的近似，通过B的低阶多项式的比率，从而导出熟悉的ARMA模型。更准确地说，$v(B)$可以写为有理分布滞后

$v(B)=\frac{\omega(B)B^b}{\delta(B)}$(12.2)

这里，分子和分母多项式定义为

$\omega(B)=\omega_0-\omega_1B-...-\omega_sB^s$

和

$\delta(B)=1-\delta_1B-...-\delta_rB^r$

$\delta(B)$的根都假定小于1。允许在x开始影响y之前可能会有b个周期的延迟的可能性，通过(12.2)中的分子分解：如果存在同期关系则b=0。



12.4冲量响应的权重$v_i$和参数$\omega_0,...,\omega_s,\delta_1,...,\delta_r$及b之间的关系总是可以通过在下式中等同$B^j$的系数来获得

$\delta(B)v(B)=\omega(B)B^b$

例如，如果r=1和s=0，则

$v(B)=\frac{\omega_0B^b}{1-\delta_1B}$

则

$v_i=0,i<b$

$v_i=\omega_0,i=b$



12.5可以假定噪声过程遵循ARMA（p，q）模型：

$n_t=\frac{\theta(B)}{\phi(B)}a_t$

这样混合传递函数噪声模型可以写成

$y_t=\frac{\omega(B)}{\delta(B)}x_{t-b}+\frac{\theta(B)}{\phi(B)}a_t$(12.3)

Box和Jenkins（1970，第11章）提出了一种识别，估计，诊断检查程序，对（12.3）形式的单输入传递函数模型。识别阶段在输出和输入之间使用互相关函数，两者使用滤波器（即ARMA模型）转换后，将$x_t$降低为白噪声，被称为预白噪声化。估计和诊断检查使用单变量对应的扩展，尽管它们不一定简单。这些年来还有其他几种识别技术，最明显的是基于使用单个滤波器（ARMA模型）预白噪声化输入和输出。



12.6如果确定单变量ARMA模型通常被认为是“艺术形式”，然后以这种方式识别传递函数就更是如此,并且，如果有多个输入，可能会变得越来越困难，因为现在的模型是：

$y_t=\sum_{j=1}^M v_j(B)x_{j,t}+n_t=\sum_{j=1}^M \frac{\omega_j(B)B^{b_j}}{\delta_j(B)}x_{j,t}+ \frac{\theta(B)}{\phi(B)}a_t$(12.4)

其中

$\omega_j(B)=\omega_{j,0}B-...-\omega_{j,s_j}B^{s_j}$

和

$\delta_j(B)=1-\delta_{j,1}B-...-\delta_{j,r_j}B^{r_j}$

最简单的方法是以“逐个”方式使用Box-Jenkins方法，确定一组单输入传递函数在y和$x_1$之间，y和$x_2$之间等，然后将它们组合以识别噪声模型，然后可以尝试进行估计和诊断检查。



## **自回归分布滞后模型**

12.8不过，如果可以制定自动模型选择程序，则很有用。尚未针对多输入模型(12.4)，但如果指定了受限形式，则此类程序变得可行。这种受限形式称为自回归分布滞后或ARDL模型，可通过对（12.4）的限制获得：

$\delta_1(B)=...=\delta_M(B)=\phi(B),\theta(B)=1$

这样，该模型，在定义$\beta_j=\omega_j(B)B^{b_j}$上并且包括截距，

$\phi(B)y_t=\beta_0+\sum_{j=1}^M\beta_j(B)x_{j,t}+a_t$(12.5)

这称为$ARDL(p,s_1,...,s_M)$模型并限制所有自回归滞后多项式相同，并且不包括滑动平均噪声分量，尽管此排除不是必需的。这些约束将噪声分量减为白噪声，通过限制动力学和OLS估计（12.5），以便选择最大滞后阶数，例如m，拟合优度统计量，例如信息准则，可以用来选择适当的设定。



12.9 ARDL表示形式（12.5）可能会以可能有用的方式重铸。回顾8.4的发展过程，每个输入多项式可能分解为

$\beta_j(B)=\beta_j(1)+\triangledown\widetilde\beta_j(B)$

其中

$\widetilde\beta_j(B)=\widetilde\beta_{j,0}+\widetilde\beta_{j,1}B+\widetilde\beta_{j,2}B^2+...+\widetilde\beta_{j,s_j-1}B^{s_j-1}$

$\widetilde\beta_{j,i}=-\sum_{l=i+1}^{s_j}\beta_{j,l}$

因此，（12.5）可以写为

$y_t=\beta_0+\sum_{i=1}^p\phi_iy_{t-i}+\sum_{j=1}^M\beta_j(1)x_{j,t}+\sum_{j=1}^M\widetilde\beta_j(B)\triangledown x_{j,t}+a_t$(12.6)

解$y_t$得

$y_t=\theta_0++\sum_{j=1}^M\theta_jx_{j,t}+\sum_{j=1}^M\widetilde\theta_j(B)\triangledown x_{j,t}+\varepsilon_t$(12.7)

其中

$\theta_0=\phi^{-1}\beta_0$

$\theta_j=\phi^{-1}(1)\beta_j(1)$

$\widetilde\theta_j(B)=\phi^{-1}(B)(\widetilde\beta_j(B)-\widetilde\phi(B)\phi^{-1}(1)\beta_j(1)),j=1,...,M$

$\varepsilon_t=\phi^{-1}(B)a_t$

其中

$\widetilde\phi(B)=\triangledown^{-1}(\phi(B)-\phi(1))$

（12.7）表示分离了短期效应的输出和输入的长期关系，但不适合直接估计。y和$x_j$之间的长期关系的估计可从(12.6)获得为

$\widetilde\theta_j=\frac{\hat\beta_j(1)}{1-\sum_{i=1}^p\hat\phi_i}=\frac{\hat\beta_{j,0}+...+\hat\beta_{j,s_j}}{1-\sum_{i=1}^p\hat\phi_i}$

可以相应地计算出伴随的标准误差。
