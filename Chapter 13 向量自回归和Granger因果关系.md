# **Chapter 13 向量自回归和Granger因果关系**

## **多元动态回归模型**

13.1 在上一章的ARDL模型的自然扩展中，假设现在有两个内生变量$y_{1,t}$和$y_{2,t}$，都与外生变量$x_t$及其滞后,以及彼此的滞后有关。在最简单的情况下，这样的模型将是：

$y_{1,t}=c_1+a_{11}y_{1,t-1}+a_{12}y_{2,t-1}+b_{10}x_t+b_{11}x_{t-1}+u_{1,t}$

$y_{2,t}=c_2+a_{21}y_{1,t-1}+a_{22}y_{2,t-1}+b_{20}x_t+b_{21}x_{t-1}+u_{2,t}$(13.1)

（13.1）等式中包含的“系统”被称为多元动态回归，在Spanos（1986）中对模型进行了较为详细的讨论。注意，“同期”变量$y_{1,t}$和$y_{2,t}$分别不包括在$y_{2,t}$和$y_{1,t}$方程式中回归项内，因为这将导致同时和识别问题，构成（13.1）的两个等式在统计上无法区分，两者都有相同的变量。当然，$y_{1,t}$和$y_{2,t}$可能是同时相关，并且任何此类相关可以进行建模，通过允许新息之间的协方差为非零，因此$E(u_{1,t}u_{2,t})=\sigma_{12}$，两个新息的方差为$E(u_1^2)=\sigma_1^2$和$E(u_2^2)=\sigma_2^2$。



13.2 （13.1）的方程对可以推广为包含n个内生变量和k个外生变量的模型。写为向量$\mathbf y^{'}_t=(y_{1,t},y_{2,t},...,y_{n,t})$和$\mathbf x^{'}_t=(x_{1,t},x_{2,t},...,x_{k,t})$，多元动态回归模型的一般形式可写为：

$\mathbf y_t=\mathbf c+\sum_{i=1}^p\mathbf A_i\mathbf y_{t-i}+\sum_{i=0}^q\mathbf B_i\mathbf x_{t-i}+\mathbf u_t$(13.2)

其中，内生变量最多有p个滞后，外生变量最多有q个滞后。这里$\mathbf c'=(c_1,c_2,...,c_n)$是$1\times n$常数向量；$\mathbf A_1,\mathbf A_2,...,\mathbf A_p$和$\mathbf B_0,\mathbf B_1,\mathbf B_2,...,\mathbf B_q$分别是$n\times n$和$n\times k$回归系数矩阵的集合，因此

$ \mathbf{A_i} = \left[ \begin{array}{ccc} a_{11,i} & a_{12,i} & \ldots & a_{1n,i}\\ a_{21,i} & a_{22,i} & \ldots & a_{2n,i}\\ \vdots & & & \vdots \\a_{n1,i} & a_{n2,i} & \ldots & a_{nn,i}\end{array} \right] ,\mathbf{B_i} = \left[ \begin{array}{ccc} b_{11,i} & b_{12,i} & \ldots & b_{1k,i}\\ b_{21,i} & b_{22,i} & \ldots & b_{2k,i}\\ \vdots & & & \vdots \\b_{n1,i} & b_{n2,i} & \ldots & b_{nk,i}\end{array} \right]$

$\mathbf u^{'}_t$是新息（或误差）的$1\times n$零均值向量，其方差和协方差可以聚集在对称误差协方差矩阵

$\mathbf \Omega=E(\mathbf u_t\mathbf u'_t)=\left[ \begin{array}{ccc} \sigma_1^2 & \sigma_{12} & \ldots & \sigma_{1n}\\ \sigma_{12} & \sigma_2^2 & \ldots & \sigma_{2n}\\ \vdots & & & \vdots \\\sigma_{1n} & \sigma_{2n} & \ldots & \sigma_n^2\end{array} \right]$

假定这些误差是相互序列不相关，因此$E(\mathbf u_{1,t}\mathbf u_{2,t})=\mathbf 0,t\ne s$，其中$\mathbf 0$是$n\times n$空矩阵。



13.3  模型（13.2）可以用（多元）最小二乘法估计，如果每个方程式中内生变量恰好有p滞后，外生变量恰好有q滞后。如果一个方程的滞后长度不同，则需要系统估计器来获得有效的估计。



## **矢量自回归**

13.4假设模型（13.2）不包含任何外生变量，使得所有的$\mathbf B_i$矩阵都为零，并且每个方程式中内生变量存在p滞后：

$\mathbf y_t=\mathbf c+\sum_{i=1}^p\mathbf A_i\mathbf y_{t-i}+\mathbf u_t$(13.3)

因为(13.3)现在只是向量$\mathbf y_t$中的p阶自回归，所以称为维度n的向量自回归（VAR（p）），并且可以用多元最小二乘估计。假设$\mathbf y_t$包含的所有序列是平稳的，这要求(13.3)相关的特征方程

$\mathbf A(B)=\mathbf I_n-\mathbf A_1B-...-\mathbf A_pB^p=\mathbf 0$

的根模数小于一（记住，np个根中某些可能显示为复共轭）。

VARs在时间序列建模多元系统方面已变得非常流行，因为没有$\mathbf x_t$项排除了必须做任何变量的内生-外生分类，这种区别通常被认为是有争议的。



## **GRANGER因果关系**

13.5 在VAR（13.3）中，在矩阵$\mathbf A_i$中存在非零非对角线元素$a_{rs,i}\ne0,r\ne s$，表示变量之间存在动态关系，否则模型将折叠为一组n个单变量AR过程。这种动态关系的存在称为Granger（-Sims）因果关系。如果$a_{rs,i}=0,i=1,2,...,p$,变量$y_s$不是变量$y_r$的Granger-因。另一方面，如果至少一个$a_{rs,i}\ne0$那么$y_s$被认为是$y_r$的Granger-因，因为这种情况下，$y_s$过去值对于预测$y_r$的当前值很有用：因此，格兰杰因果关系是“预测”的标准。如果$y_r$也是$y_s$的Granger-因，这对变量称为表现出反馈。



13.6在VAR（p）中，格兰杰因果关系从$y_s$到$y_r$，可以描述为$y_s\to y_r$，可以通过建立非格兰杰因果关系（$y_s不\to y_r$）的原假设来估计，$H_0:a_{rs,1}=...=a_{rs,p}=0$，用Wald统计量对此进行检验。



## **确定矢量自回归的滞后阶数**

13.8 为使VAR投入运行，滞后阶数p，通常是未知的，需要根据经验确定。选择滞后阶数的传统方式是使用阶数检验程序。考虑模型（13.3）与误差协方差矩阵$\mathbf \Omega_p=E(\mathbf u_t\mathbf u'_t)$，其中包含p下标以强调矩阵与VAR（p）有关。该矩阵的一个估计为：

$\mathbf{\hat\Omega}_p=(T-p)^{-1}\mathbf {\hat U}_p\mathbf {\hat U}'_p$

其中$\mathbf {\hat U}_p=(\mathbf {\hat u}_{p,1}',...,\mathbf {\hat u}_{p,n}')'$是通过VAR（p）的OLS估计获得的残差矩阵，$\mathbf {\hat u}_{p,r}=( {\hat u}_{r,p+1},...,{\hat u}_{r,T})'$是来自第r个方程（注意，对于样本数量T，p个观测值将通过滞后而丢失）的残差向量。用于检验阶数p，反对m阶，m\<p的似然比（LR）统计量，是

$LR(p,m)=(T-np)log(\frac{|\mathbf{\hat\Omega}_m|}{|\mathbf{\hat\Omega}_p|})\sim\chi_{n^2(p-m)}^2$(13.4)

因此，如果LR（p，m)超过自由度$n^2(p-m)$的$\chi^2$分布的α临界值，则VAR阶为m的假设在α的显著水平上被拒绝，支持更高阶p。统计量使用比例因子T-np而不是T-p来解释可能的小样本偏差。

可以依次使用统计量（13.4）由p的最大值$p_{max}$开始，例如，先检验$p_{max}$，反对$p_{max}-1$，使用$LR(p_{max},p_{max}-1)$，并且如果该统计量不显著，则检验$p_{max}-1$反对$p_{max}-2$，使用$LR(p_{max}-1,p_{max}-2)$，一直持续到获得显著检验。

可替代地，可以将某种类型的信息标准最小化。例如，多元AIC和BIC标准定义为：

$MAIC(p)=log(|\mathbf{\hat\Omega}_p|)+(2+n^2p)T^{-1}$

$MBIC(p)=log(|\mathbf{\hat\Omega}_p|)+n^2pT^{-1}lnT,p=0,1,...,p_{max}$



## 方差分解与新息解释

13.10虽然VAR（1）的估计系数相对容易解释，这对于高阶VAR很快成为问题，因为不仅系数的数量迅速增加（每个附加的滞后引入了另外$n^2$个系数），而且其中许多系数估计不精确且高度相关，因此在统计上变得不显著。



13.11这导致了几种方法的发展，来检查基于$y_t$的矢量滑动平均表示（VMA）的VAR的“信息内容”。假设VAR写为滞后算子形式

$\mathbf A(B)\mathbf y_t=\mathbf u_t$

其中，如在13.4中

$\mathbf A(B)=\mathbf I_n-\mathbf A_1B-...-\mathbf A_pB^p$

是B中的矩阵多项式。类似于单变量情况，（无限阶）VMA表示为

$\mathbf y_t=\mathbf A^{-1}(B)\mathbf u_t=\mathbf \Psi(B)\mathbf u_t=\mathbf u_t+\sum_{i=1}^\infty\mathbf \Psi_i\mathbf u_{t-i}$(13.5)

其中

$\mathbf \Psi_i=\sum_{j=1}^i\mathbf A_j\mathbf \Psi_{i-j},\mathbf \Psi_0=\mathbf I_n,\mathbf \Psi_i=\mathbf 0,i<0$

通过在$\mathbf \Psi(B)\mathbf A(B)=\mathbf I_n$等同B的系数而获得这种递推。



13.12  $\mathbf \Psi_i$矩阵可以被解释为系统的动态乘数，因为它们代表模型对每个变量中单位冲击的响应。$y_r$对$y_s$的单位冲击的响应（由$u_{s,t}$产生；取值1而不是其期望值零），由脉冲响应函数给出，是序列$\psi_{rs,1},\psi_{rs,2},...$，其中$\psi_{rs,i}$是矩阵$\mathbf \Psi_i$的第rs元。

既然$\mathbf \Omega_p=E(\mathbf u_t\mathbf u'_t)$不需要是对角线的，$\mathbf u_t$的分量可以同时相关。如果这些相关性很高，模拟到$y_s$的冲击，而$\mathbf u_t$的所有其它分量保持恒定，可能会产生误导，因为没有办法分离$y_r$对$y_s$冲击的响应，从其对其他与$u_{s,t}$相关的冲击的响应中。但是，如果我们定义下三角矩阵$\mathbf S$，则$\mathbf S\mathbf S'=\mathbf \Omega_p$,定义$\mathbf v_t=\mathbf S^{-1}\mathbf u_t$，则$E(\mathbf v_t\mathbf v_t')=\mathbf I_n$,转化误差$\mathbf v_t$彼此正交（这称为Cholesky分解）。然后可以将VMA表示重新归一化为递归形式：

$\mathbf y_t=\sum_{i=0}^\infty(\mathbf \Psi_i\mathbf S)(\mathbf S^{-1}\mathbf u_{t-i})=\sum_{i=0}^\infty\mathbf \Psi_i^O\mathbf v_{t-i}$

其中$\mathbf \Psi_i^O=\mathbf \Phi_i\mathbf S$（使得$\mathbf \Psi_0^O=\mathbf \Phi_0\mathbf S$为下三角）。$y_r$对$y_s$脉冲响应函数由序列$\psi_{rs,0}^O,\psi_{rs,1}^O,\psi_{rs,2}^O,...$给出，每个脉冲响应可以简洁写为：

$\psi_{rs,i}^O=\mathbf e_r'\mathbf \Psi_i\mathbf S\mathbf e_s$(13.6)

这里$\mathbf e_s$是$n\times1$选择向量，第s元素为1，其他为0。此序列称为正交脉冲响应函数。这样，（累积的）长期响应是：

$\psi_{rs}^O(\infty)=\sum_{i=0}^\infty\mathbf e_r'\mathbf \Psi_i\mathbf S\mathbf e_s$(13.7)

然后，可以将整个长期响应集合在矩阵

$\mathbf \Psi^O(\infty)=\sum_{i=0}^\infty\mathbf \Psi_i\mathbf S=\mathbf \Psi(1)\mathbf S$



13.13  $\mathbf v_t$s的不相关性允许$y_r$的h步超前预测误差方差将分解为新息“核算”部分，一种称为新息核算的技术，是Sims（1981）创造的术语。例如，$y_r$对$y_s$的h步预测误差方差，由正交新息核算的比例，由下式给出：

$V_{rs,h}^O=\frac{\sum_{i=0}^h(\psi_{rs,h}^O)^2}{\sum_{i=0}^h\mathbf e_r'\mathbf \Psi_i\mathbf \Omega_p\mathbf \Psi_i'\mathbf e_r}=\frac{\sum_{i=0}^h(\mathbf e_r'\mathbf \Psi_i\mathbf S\mathbf e_s)^2}{\sum_{i=0}^h\mathbf e_r'\mathbf \Psi_i\mathbf \Omega_p\mathbf \Psi_i'\mathbf e_r}$

对于较大的h，此正交化的预测误差方差分解允许隔离那些对可变性的相对贡献，从直觉上讲，是“持久的”。

但是，正交化技术确实具有重要缺点，对于$\mathbf S$矩阵的选择不是唯一的，因此不同变量的组合会改变系数$\psi_{rs,i}^O$，因此改变脉冲响应函数和方差分解。这些变化的程度将取决于新息的同期相关性的大小。



13.14 除了比较变量其他组合的脉冲响应和方差分解，解决此问题的一种方法是Pesaran和Shin（1997）的广义脉冲响应，通过在（13.6）更换$\mathbf S$,用$\sigma_r^{-1}\mathbf \Omega_p$定义：

$\psi_{rs,i}^G=\sigma_r^{-1}\mathbf e_r'\mathbf \Psi_i\mathbf \Omega_p\mathbf e_s$

广义的脉冲响应对变量组合不变，是唯一的，并充分说明了在不同的冲击中观测的相关性的过去模式。正交和广义脉冲响应一致，只有当$\mathbf \Omega_p$是对角的，而在一般情况下，仅对s=1相同。



## **结构矢量自回归**

13.15 VAR的“非不变性”产生了对方差分解方法非常详细的分析和批评，主要着重于在传统的计量经济学意义上无法将VAR视为“结构性”，因此冲击不能用特定变量识别，除非事先做出识别假设，否则计算出的脉冲响应函数和方差分解将无效。$\mathbf S$的三角形“递归”结构因无理论而被赞誉，并导致了其他识别限制的发展，基于理论考虑，使用结构VAR（SVAR）方法：参阅Cooley和LeRoy（1985）；Blanchard（1989）；Blanchard和Quah（1989）。



13.16  13.12的Cholesky分解可以写成$\mathbf u_t=\mathbf S\mathbf v_t$，且$\mathbf S\mathbf S'=\mathbf \Omega_p$,$E(\mathbf v_t\mathbf v_t')=\mathbf I_n$。一个更一般的表述是：

$\mathbf A\mathbf u_t=\mathbf B\mathbf v_t$

因此

$\mathbf B\mathbf B'=\mathbf A\mathbf \Omega_p\mathbf A'$(13.8)

由于$\mathbf A$和$\mathbf B$都是$n\times n$矩阵，所以它们包含$2n^2$个元素，但是矩阵的对称性对（13.8）的两侧都施加$n(n+1)/2$个限制。另外至少施加$2n^2-n(n+1)/2$个限制，以完成对$\mathbf A$和$\mathbf B$的识别。通常是一些元素的特定值：例如，如果n=3则定义

$\mathbf A=\left[\begin{array}{} a_{11} &0 &0 \\a_{21} &a_{22}&0\\a_{31}&a_{32}&a_{33} \end{array}\right],\mathbf B=\left[\begin{array}{} 1 &0 &0 \\0 &1 &0\\0 &0 &1\end{array}\right]$

施加了12个限制来获取Cholesky分解所需的形式。同样，具有

$\mathbf A=\left[\begin{array}{} 1 &0 &0 \\a_{21} &1&0\\a_{31}&a_{32}&1 \end{array}\right],\mathbf B=\left[\begin{array}{} b_{11} &0 &0 \\0 & b_{22} &0\\0 &0 & b_{33}\end{array}\right]$

的系统也将被识别，$\mathbf B$对角线上的系数给出了“非标准化”结构新息的标准偏差。



13.17 也可以使用另一种形式的限制。长期脉冲响应可以推广（13.7）写为

$\psi_{rs}^O(\infty)=\sum_{i=0}^\infty\mathbf e_r'\mathbf \Psi_i\mathbf A^{-1}\mathbf B\mathbf e_s$

或者，矩阵形式

$\mathbf \Psi(\infty)=\sum_{i=0}^\infty\mathbf \Psi_i\mathbf A^{-1}\mathbf B=\mathbf \Psi(1)\mathbf A^{-1}\mathbf B$

可能对$\mathbf \Psi(\infty)$元素限制，通常他们取零：例如，设置$\psi_{rs}(\infty)=0$，限制$y_r$对$y_s$冲击的长期响应为零。
