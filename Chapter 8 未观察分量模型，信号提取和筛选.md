# Chapter 8 未观测分量模型，信号提取和滤波

## 未观测分量模型

8.1 一个差分平稳，即$I(1)$，时间序列可能总是分解为随机的非平稳趋势或信号分量和平稳噪声或不规则分量：

$x_t=z_t+\mu_t$(8.1)

可以以几种方式执行这种分解。例如，Muth（1960）的经典示例假设趋势分量$z_t$是随机游走

$z_t=\mu+z_{t-1}+v_t$

$\mu_t$是白噪声且与$v_t$独立，即$u_t\sim WN(0,\sigma_{u}^2),v_t\sim WN(0,\sigma_{v}^2)$,对所有i,$E(u_tv_{t-i})=0$。因此，$\triangledown x_t$是平稳过程

$\triangledown x_t=\mu+v_t+u_t-u_{t-1}$(8.2)

具有在滞后一处中断的自相关函数，系数为

$\rho_1=-\frac{\sigma_u^2}{\sigma_u^2+2\sigma_v^2}$(8.3)

很明显$-0.5\le\rho_1\le0$，精确值取决于两个方差的相关大小，因此$\triangledown x_t$可以写成MA（1）过程

$\triangledown x_t=\mu+e_t-\theta e_{t-1}$(8.4)

其中$e_t\sim WN(0,\sigma_{e}^2)$。定义$\kappa=\sigma_v^2/\sigma_u^2$为信噪方差比率(signal-to-noise variance ratio)，(8.2)与(8.4)参数之间的关系可以是

$\theta=1/2((\kappa+2)-(\kappa^2+4\kappa)^{1/2}),\kappa=\frac{(1-\theta)^2}{\theta},\kappa\ge0,|\theta|<1$

和

$\sigma_u^2=\theta \sigma_e^2$

因此，κ=0对应于θ=1，使得(8.4)的单位根“抵消”且过度差分$x_t$是平稳的，而$\kappa=\infty$对应于θ=0，在这种情况下，$x_t$是纯随机游动。平稳零假设检验θ=1，也被认为是零假设$\sigma_v^2=0$的检验，因为如果是这种情况，则$z_t$是确定性线性趋势。



8.2  (8.1)形式的模型被称为未观测分量（UC）模型，这些分量的更一般的表述是：

$\triangledown z_t=\mu+\gamma(B)v_t,u_t=\lambda(B)a_t$(8.5)

其中$v_t$和$a_t$是具有有限方差$\sigma_v^2$和$\sigma_a^2$的独立白噪声序列，并且其中$\gamma(B)$和$\lambda(B)$是没有共同根的平稳多项式。可以证明$x_t$将具有以下形式：

$\triangledown x_t=\mu+\theta(B)e_t$(8.6)

其中$\theta(B)$和$\sigma_e^2$可以从下式获得：

$\sigma_e^2\frac{\theta(B)\theta(B^{-1})}{(1-B)(1-B^{-1})}=\sigma_v^2\frac{\gamma(B)\gamma(B^{-1})}{(1-B)(1-B^{-1})}+\sigma_a^2\lambda(B)\lambda(B^{-1})$(8.7)

由此我们可以看出，不必要只从(8.6)的参数中识别出分量的参数，实际上，通常不会识别分量。但是，如果$z_t$被限制为一个随机游走（$\gamma(B)=1$），则UC模型的参数将被识别。对于Muth模型来说显然是这样，既然$\sigma_u^2$可以通过$\triangledown x_t$的滞后一自协方差(（8.3)的分子）估计，$\sigma_v^2$可以通过$\triangledown x_t$的方差(（8.3)的分子），和$\sigma_u^2$的估计值来估计。



8.3然而，这个例子说明，即使方差确定，这样的分解可能并不总是可行的，因为它无法在$\triangledown x_t$中考虑正一阶自相关。为此，需要放宽$z_t$是随机游走的假设，使趋势成分包含永久性和暂时性运动，或者假设$v_t$和$a_t$是独立的。如果这些假设之一放宽，将不会识别Muth模型的参数。



8.4趋势分量$z_t$遵循随机游走的假设乍一看似乎没有那么严格。考虑$\triangledown x_t$的Wold分解：

$\triangledown x_t=\mu+\psi(B)e_t=\mu+\sum_{j=0}^{\infty}\psi_je_{t-j}$(8.8)

由于$\psi(1)=\sum\psi_j$是一个常数，我们可以写成：

$\psi(B)=\psi(1)+C(B)$

因此：

$C(B)=\psi(B)-\psi(1)=(1-B)(-\psi_1-\psi_2(1+B)-\psi_3(1+B+B^2)-...)$

即，

$C(B)=(1-B)(-(\sum_{j=1}^{\infty}\psi_j)-(\sum_{j=2}^{\infty}\psi_j)B-(\sum_{j=3}^{\infty}\psi_j)B^2-...)=\triangledown \widetilde\psi(B)$

因此，

$\psi(B)=\psi(1)+\triangledown \widetilde\psi(B)$

说明：

$\triangledown x_t=\mu+\psi(1)e_t+\triangledown \widetilde\psi(B)e_t$

这给出了贝弗里奇和尼尔森（1981）的分解，分量

$\triangledown z_t=\mu+(\sum_{j=0}^{\infty}\psi_j)e_t=\mu+\psi(1)e_t$

和

$u_t=-(\sum_{j=1}^{\infty}\psi_j)e_t-(\sum_{j=2}^{\infty}\psi_j)e_{t-1}-(\sum_{j=3}^{\infty}\psi_j)e_{t-2}-...=\widetilde\psi(B)e_t$

由于$e_t$是白噪声，因此趋势分量是随机游动，漂移速率μ加一个新息$\psi(1)e_t$，因此与原始序列的成比例。噪声分量是显然是平稳的，但是由于它是由与趋势分量相同的新息驱动的，$z_t$和$u_t$必须完全相关，这与Muth分解假设它们是独立的形成直接对比。例如，ARIMA（0,1,1）过程(8.4)的Beveridge-Nelson分解为：

$\triangledown z_t=\mu+(1-\theta)e_t$(8.10)

$u_t=\theta e_t$(8.11)



8.5 Beveridge-Nelson分解和Muth分解之间的关系是准确的。不假设$u_t$和$v_t$是独立的，而是$v_t=\alpha u_t$。等式(8.2)和(8.4)得到：

$\triangledown x_t=\mu+(1+\alpha)u_t-u_{t-1}=\mu+e_t-\theta e_{t-1}$

因此$e_t=(1+\alpha)u_t$和$\theta e_t=u_t$，因此恢复（8.11）并说明$\theta=1/(1+\alpha)$。趋势(8.10)变为：

$\triangledown z_t=\mu+(1-\theta)e_t=\mu+\frac{1-\theta}{\theta}u_t=\mu+\alpha u_t=\mu+v_t$

恢复了Muth趋势。



8.6 遵循Newbold（1990）的一种直接估计Beveridge-Nelson分量的方法是近似Wold分解(8.8)，通过设置$\psi(B)=\theta(B)/\phi(B)$的ARIMA（P，1，q）过程：

$\triangledown x_t=\mu+\frac{\theta(B)}{\phi(B)}e_t=\mu+\frac{1-\theta_1B-...-\theta_qB^q}{1-\phi_1B-...-\phi_pB^p}e_t$(8.12)

因此

$\triangledown z_t=\mu+\psi(1)e_t=\mu+\frac{\theta(1)}{\phi(1)}e_t=\mu+\frac{1-\theta_1-...-\theta_q}{1-\phi_1-...-\phi_p}e_t$(8.13)

等式(8.12)也可以写成

$\frac{\phi(B)}{\theta(B)}\psi(1)\triangledown x_t=\mu+\psi(1)e_t$

比较(8.13)和(8.14)发现

$z_t=\frac{\phi(B)}{\theta(B)}\psi(1) x_t=\omega(B)x_t$

因此，趋势是观测序列当前和过去值的加权平均值。权重的总和为1，既然$\omega(1)=1$。噪声分量由下式给出：

$u_t=x_t-\omega(B)x_t=(1-\omega(B))x_t=\widetilde \omega(B)x_t=\frac{\phi(1)\theta(B)-\phi(B)\theta(1)}{\phi(1)\theta(B)}x_t$

既然$\widetilde \omega(1)=1-\omega(1)=0$，噪声分量的权重总和为零。使用(8.12)，此成分也可以表示为

$u_t=\frac{\phi(1)\theta(B)-\phi(B)\theta(1)}{\phi(1)\phi(B)\triangledown}e_t$(8.15)

因为$u_t$平稳，(8.15)的分子可以写成$\phi(1)\theta(B)-\phi(B)\theta(1)=\triangledown\varphi(B)$，既然它必须包含一个单位根来抵消分母中的。由于分子的阶为$max(p,q)$，$\varphi(B)$必须为$r=max(p,q)-1$阶；说明噪音有ARMA（p，r）表示

$\phi(B)u_t=\frac{\varphi(B)}{\phi(1)}e_t$

例如，对于ARIMA（0,1,1）过程(8.4),分量为：

$z_t=(1-\theta B)^{-1}(1-\theta)x_t=(1-\theta)\sum_{j=0}^{\infty}\theta^jx_{t-j}$

和

$u_t=\frac{(1-\theta B)-(1-\theta)}{(1-\theta B)}x_t=\frac{\theta (1-B)}{(1-\theta B)}x_t=\theta(1-\theta B)^{-1}\triangledown x_t=\theta\sum_{j=0}^{\infty}\theta^jx_{t-j}$

因此，可以将趋势递归估计为：

$\hat z_t=\theta\hat z_{t-1}+(1-\theta)x_t,\hat u_t=x_t-\hat z_t$

起始值为$\hat z_1=x_1,\hat u_1=0$。



## 信号提取

8.8 给定(8.1)形式的UC模型以及$z_t$和$u_t$的模型，通常有助于估计这两个未观测成分，一个称为信号提取的程序。$z_t$的MMSE估计是使$E(\zeta_t^2)$最小化估计$\hat z_t$，其中$\zeta_t=zt-\hat z_t$是估计误差。Pierce（ 1979） 表示给定无穷样本观测$x_t,-\infty \leq t \leq\infty$,这样的估计量是

$\hat z_t=v_z(B)x_t=\sum_{j=-\infty}^{\infty}v_{zj}x_{t-j}$

其中滤波器$v_z(B)$定义为：

$v_z(B)=\frac{\sigma_v^2\gamma(B)\gamma(B^{-1})}{\sigma_v^2\theta(B)\theta(B^{-1})}$

在这种情况下，噪声分量可以估计为：

$\hat u_t=x_t-\hat z_t=(1-v_z(B))x_t=v_u(B)x_t$

例如，对于覆盖白噪声的随机游走的Muth模型：

$v_z(B)=\frac{\sigma_v^2}{\sigma_e^2}(1-\theta B)^{-1}(1-\theta B^{-1})^{-1}=\frac{\sigma_v^2}{\sigma_e^2}\frac{1}{(1-\theta^2)}\sum_{j=-\infty}^{\infty}\theta^{|j|}B^j$

所以，用(8.6)得到$\sigma_v^2=(1-\theta)^2\sigma_e^2$，我们有：

$\hat z_t=\frac{(1-\theta)^2}{(1-\theta^2)}\sum_{j=-\infty}^{\infty}\theta^{|j|}x_{t-j}$

因此，对于θ值接近一，$\hat z_t$将由x的未来值和过去值的极长的滑动动平均值给定。但是，如果θ接近零，几乎等于x的最近观测值。从（8.3），较大的θ值对应于较小的信噪方差比$\kappa=\sigma_v^2/\sigma_u^2$。当噪声分量占主导地位时，x值的长期滑动平均将提供趋势的最佳估计，如果噪声分量很小，那么趋势就可以由x的当前位置决定。



8.9估计误差可写为：

$\zeta_t=z_t-\hat z_t=v_z(B)z_t-v_u(B)u_t$

皮尔斯（1979）表示，如果$z_t$和$u_t$由（8.4）形式的过程生成，将平稳。事实上，$\zeta_t$将遵循过程：

$\zeta_t=\theta_{\zeta}(B)\xi_t$

其中

$\theta_{\zeta}(B)=\frac{\gamma(B)\lambda(B)}{\theta(B)},\sigma_{\xi}^2=\frac{\sigma_a^2\sigma_v^2}{\sigma_e^2}$

且$\xi_t\sim WN(0,\sigma_{\xi}^2)$。

Muth模型$\zeta_t$遵循AR（1）过程

$(1-\theta B)\zeta_t=\xi_t$

且最佳信号提取程序的均方误差为：

$E(\zeta_t^2)=\frac{\sigma_a^2\sigma_v^2}{\sigma_e^2(1-\theta^2)}$



8.10 如前所述，如果仅给出$x_t$及其模型的实现，即(8.6），则通常无法确定$z_t$和$u_t$的分量模型。如果$x_t$遵循ARIMA（0,1,1）过程

$\triangledown x_t=(1-\theta B)e_t$(8.16)

那么最一般的"信号加白噪声"UC模型的$z_t$为：

$\triangledown z_t=(1-\Theta B)v_t$(8.17)

对于区间$-1\leq\Theta\leq\theta$中的任何Θ值，都存在$\sigma_a^2$和$\sigma_v^2$使$z_t+u_t$产生(8.16)。可以看出设置Θ=-1最小化$z_t$和$u_t$的方差，这被称为$x_t$的典范分解。选择此值表示$\gamma(B)=1+B$，我们有：

$\hat z_t=\frac{\sigma_v^2(1+B)(1+B^{-1})}{\sigma_e^2(1-\theta B)(1-\theta B^{-1})}$

和

$(1-\theta B)\zeta_t=(1+B)\xi_t$



## 滤波器

8.13 UC模型（8.5）也与Hodrick-Prescott趋势滤波器（Hodrick和Prescott，1997年）有关，这是经济学时间序列去趋势的一种流行方法。通过最小化噪声分量$u_t=x_t-z_t$的变化得出该滤波器，受制于趋势分量$z_t$的“平滑度”。这种平滑状态不利于趋势的加速，因此最小化问题变成了最小化如下函数：

$\sum_{t=1}^Tu_t^2+\lambda\sum_{t=1}^T((z_{t+1}-z_t)-(z_t-z_{t-1}))^2$

关于$z_t,t=1,2,...,T+1$，其中λ是拉格朗日乘数，可以解释为平滑度参数。λ的值越高，趋势越平滑，因此在极限情况下，随着$\lambda\to\le$，$z_t$变为线性趋势。一阶条件为：

$0=-2(x_t-z_t)+2\lambda((z_t-z_{t-1})-(z_{t-1}-z_{t-2}))-4\lambda((z_{t+1}-z_t)-(z_t-z_{t-1}))+2\lambda((z_{t+2}-z_{t+1})-(z_{t+1}-z_t))$

可以写成：

$x_t=z_t+\lambda(1-B)^2(z_t-2z_{t+1}+z_{t+2})=(1+\lambda(1-B)^2(1-B^{-1})^2)z_t$

因此Hodrick-Prescott（H-P）趋势估计为

$\hat z_t(\lambda)=(1+\lambda(1-B)^2(1-B^{-1})^2)^{-1}x_t$(8.18)

MMSE趋势估计可以用(8.7)写为：

$\hat z_t=\frac{\sigma_v^2\gamma(B)\gamma(B^{-1})}{\sigma_e^2\theta(B)\theta(B^{-1})}x_t=\frac{\gamma(B)\gamma(B^{-1})}{\gamma(B)\gamma(B^{-1})+(\sigma_a^2/\sigma_v^2)\lambda(B)\lambda(B^{-1})}x_t$

将其与H-P趋势估计(8.18)比较表明，为了后者要在MMSE方面达到最佳，我们必须设定

$\gamma(B)=(1-B)^{-1},\lambda(B)=1,\delta=\frac{\sigma_a^2}{\sigma_v^2}$

换句话说，基本的UC模型必须具有趋势分量$\triangledown^2z_t=v_t$且$u_t$必须是白噪声。
