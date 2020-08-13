# Chapter 15 有单积变量的向量自回归，向量误差校正模型，以及共同趋势

## 单积变量的向量自回归

15.1 已经分析了$I(1)$变量的影响，协整的可能，在单方程自回归分布滞后模型，允许向量自回归包含$I(1)$变量的暗含显然需要讨论。然后，考虑13.11的n变量VAR（p），

$\boldsymbol A(B)\boldsymbol y_t=\boldsymbol c+\boldsymbol u_t$(15.1)

其中，在13.2，$E(\boldsymbol u_t)=0$,且

$E(\boldsymbol u_t\boldsymbol u_s)=\left\{ \begin{array}{} \mathbf \Omega & t=s\\ \mathbf 0 & t\ne s\end{array} \right.$

使用8.4“Beveridge-Nelson”分解的矩阵推广，写出矩阵多项式

$\boldsymbol A(B)=\boldsymbol I_n-\sum_{i=1}^p\boldsymbol  A_i B^i$

对p\>1，为

$\boldsymbol A(B)=(\boldsymbol I_n-\boldsymbol AB)-\boldsymbol  \Phi(B) B\triangledown$

其中

$\boldsymbol A=\sum_{i=1}^p\boldsymbol  A_i$

和

$\boldsymbol \Phi(B)=\sum_{i=1}^{p-1}\boldsymbol \Phi_iB^{i-1},\boldsymbol \Phi_i=-\sum_{j=i+1}^p\boldsymbol  A_j$

$\boldsymbol \Phi_i$矩阵可以由递归$\boldsymbol \Phi_1=-\boldsymbol A+\boldsymbol A_1$,$\boldsymbol \Phi_i=\boldsymbol \Phi_{i-1}+\boldsymbol A_i,i=2,...,p-1$获得。随着A（B）的分解，（15.1）可以总是写成

$(\boldsymbol I_n-\boldsymbol AB-\boldsymbol \Phi(B)\triangledown)\boldsymbol y_t=\boldsymbol c+\boldsymbol u_t$

或

$\boldsymbol y_t=\boldsymbol c+\boldsymbol \Phi(B)\triangledown\boldsymbol y_{t-1}+\boldsymbol A\boldsymbol y_{t-1}+\boldsymbol u_t$

等效表示为

$\triangledown\boldsymbol y_t=\boldsymbol c+\boldsymbol \Phi(\mathbf B)\triangledown\boldsymbol y_{t-1}+\boldsymbol \Pi\boldsymbol y_{t-1}+\boldsymbol u_t$(15.2)

其中

$\boldsymbol \Pi=\boldsymbol A-\boldsymbol I_n=-\boldsymbol A(1)$

称为长期矩阵。表示（15.2）是对应于ADF回归的多变量对应，并且应强调这是（15.1）的纯代数变换，因为至此，没有关于$\boldsymbol y_t$性质的假设。



15.2首先考虑$\boldsymbol A=\boldsymbol I_n$的情况，使得$\boldsymbol \Pi=\boldsymbol0$和$\triangledown\boldsymbol y_t$服从VAR（p-1）

$\triangledown\boldsymbol y_t=\boldsymbol c+\boldsymbol \Phi(\mathbf B)\triangledown\boldsymbol y_{t-1}+\boldsymbol u_t$(15.3)

其中$\boldsymbol y_t$是一个$I(1)$过程，以及一阶差分$\triangledown\boldsymbol y_t$的VAR，如（15.3），是合适的设定。



15.3以下是等式（15.1）和（15.3）之间存在的关系一些有趣而重要的结果。从15.1的递归可以看出的系数矩阵（15.1）和（15.2）之间存在直接联系：

$\boldsymbol A_p=-\boldsymbol \Phi_{p-1}$

$\boldsymbol A_i=\boldsymbol \Phi_i-\boldsymbol \Phi_{i-1},i=2,...,p-1$

$\boldsymbol A_1=\boldsymbol \Pi+\boldsymbol I_n+\boldsymbol \Phi_1$

然后可以看出（例如，参见Hamilton，1994）不考虑$\boldsymbol y_t$的单积阶数，$\boldsymbol A_i$单个系数的t检验是渐近有效的，因为是$\boldsymbol A_i$的线性组合的F检验，除“单位根”组合$\boldsymbol A_1+\boldsymbol A_2+...+\boldsymbol A_p$以外。确定滞后阶数的似然比检验也渐近有效，如使用信息标准。但是，无效的是Granger因果检验，其变成不具有通常的$\chi^2$极限分布。这些结果不管中$\boldsymbol y_t$的变量是否有漂移都成立。



## 向量自回归与协整变量

15.4条件$\boldsymbol A=\boldsymbol I_n$意味着

$|\boldsymbol \Pi|=|\boldsymbol A_1+...+\boldsymbol A_p-\boldsymbol I_n|=0$(15.4)

也就是说，长期矩阵是奇异的，因此必须秩小于n。然后说VAR（15.1）包含至少一个单元根。但是注意，（15.4）不一定意味着$\boldsymbol A=\boldsymbol I_n$,正是这一事实导致了协整VAR（CVAR）。因此，假设（15.4）成立，从而使长期矩阵$\boldsymbol \Pi$是奇异的，且$|\boldsymbol \Pi|=0$，但是$\boldsymbol \Pi\ne \boldsymbol 0$,$\boldsymbol A\ne\boldsymbol I_n$。由于奇异，$\boldsymbol \Pi$将降低秩，例如r，其中$0\le r\le n$。在这种情况下，$\boldsymbol \Pi$可表示为两个$n\times r$矩阵$\boldsymbol \alpha$和$\boldsymbol \beta$的乘积，使得$\boldsymbol \Pi=\boldsymbol \beta\boldsymbol \alpha'$。

要了解为什么会这样，注意$\boldsymbol \alpha'$可以定义为包含$\boldsymbol \Pi$线性依赖的r行的矩阵，使$\boldsymbol \Pi$必须能够被写成$\boldsymbol \alpha'$的线性组合：$\boldsymbol \beta$是需要的系数矩阵。这些$\boldsymbol \Pi$的线性无关的r行，写为$\boldsymbol \alpha'=(\boldsymbol \alpha_1,...,\boldsymbol \alpha_r)$，称为协整向量，$\boldsymbol \Pi$将仅包含n-r个单位根，而不是n个单位根如果$\boldsymbol \Pi=\boldsymbol0 $，r=0为这种情况。



15.5为什么$\boldsymbol \alpha'$的行被称为协整向量？$\boldsymbol \Pi=\boldsymbol \beta\boldsymbol \alpha'$代入（15.2）得到

$\triangledown\boldsymbol y_t=\boldsymbol c+\boldsymbol \Phi(\mathbf B)\triangledown\boldsymbol y_{t-1}+\boldsymbol \beta\boldsymbol \alpha'\boldsymbol y_{t-1}+\boldsymbol u_t$(15.5)

假设$\boldsymbol y_t\sim I(1)$意味着，$\triangledown\boldsymbol y_t\sim I(0)$，必须是$\boldsymbol A_1=\boldsymbol \Pi+\boldsymbol I_n+\boldsymbol \Phi_1$的情况以确保（15.5）的两侧“平衡”，也就是说，它们具有相同的单积阶数。换句话说，$\boldsymbol \alpha'$是一个矩阵当其行乘以$\boldsymbol y_t$时，会得到$\boldsymbol y_t$平稳的线性组合：r个线性组合$\boldsymbol \alpha_1'\boldsymbol y_t,...,\boldsymbol \alpha_r'\boldsymbol y_t$都是平稳的，因此可以起到协整关系的作用。



15.6因此，如果$\boldsymbol y_t$协整，协整秩为r，则它可以表示为向量误差校正模型（VECM）

$\triangledown\boldsymbol y_t=\boldsymbol c+\boldsymbol \Phi(\mathbf B)\triangledown\boldsymbol y_{t-1}+\boldsymbol \beta\boldsymbol e_{t-1}+\boldsymbol u_t$(15.6)

其中$\boldsymbol e_{t-1}=\boldsymbol \alpha'\boldsymbol y_t$包含r个平稳误差校正。这被称为Granger表示定理，显然是（14.4）的多元扩展和推广。



15.7还有几点值得一提。参数矩阵$\boldsymbol \alpha$和$\boldsymbol \beta$不能唯一识别，因为对于任何非奇异的$n\times n$矩阵$\xi$，乘积$\boldsymbol \beta\boldsymbol \alpha'$和$\boldsymbol \beta\boldsymbol \xi(\boldsymbol \xi^{-1}\boldsymbol \alpha')$都等于$\boldsymbol \Pi$。,因此通常会施加一些标准化，首选是将$\boldsymbol \alpha$的某些元素设置为1。

如果r=0，那么我们已经在15.2中看到该模型在一阶差分$\triangledown\boldsymbol y_t$变为VAR(p-1)（15.3）。另一方面，如果r=n，则$\boldsymbol \Pi$满秩且非奇异，并且$\boldsymbol y_t$将不包含单位根，将为$I(0)$，因此$\boldsymbol y_t$水平的VAR(p)从一开始就合适。

误差校正$\boldsymbol e_t$尽管是平稳的，但不限制有零均值，如（15.6）所示，$\boldsymbol y_t$的增长可以通过误差校正$\boldsymbol e_t$和“自主”漂移分量$\boldsymbol c$。怎么在（15.6）中处理这个截距，或许是一个趋势，对于确定适当的估计程序和推断使用的一组临界值很重要。



## **向量误差校正模型的估计和协整秩的检验**

15.8 VECM（15.5）的估计是非标准的，因为$\boldsymbol \alpha$和$\boldsymbol \beta$矩阵以乘积$\boldsymbol \beta\boldsymbol \alpha'$进入非线性。不用不必要的技术细节，以下将获得ML估计。再次考虑（15.5），但现在写为：

$\triangledown\boldsymbol y_t=\boldsymbol c+\sum_{i=1}^{p-1}\boldsymbol \Phi_i\triangledown\boldsymbol y_{t-i}+\boldsymbol \beta\boldsymbol \alpha'\boldsymbol y_{t-1}+\boldsymbol u_t$(15.7)

第一步是在约束$\boldsymbol \beta\boldsymbol \alpha'=\boldsymbol0$下估计（15.7）。这样就是“差分VAR”（15.3），OLS估计将得出残差$\hat{\boldsymbol {u}}_t$的集合，我们可以据此计算样本协方差矩阵

$\boldsymbol S_{00}=T^{-1}\sum_{t=1}^T\hat{\boldsymbol {u}}_t\hat{\boldsymbol {u}}_t'$

第二步是用OLS估计多元回归

$\boldsymbol y_{t-1}=\boldsymbol d+\sum_{i=1}^{p-1}\boldsymbol \Xi\triangledown \boldsymbol y_{t-i}+\boldsymbol v_t$

并使用残差$\boldsymbol v_t$计算协方差矩阵

$\boldsymbol S_{11}=T^{-1}\sum_{t=1}^T\hat{\boldsymbol {v}}_t\hat{\boldsymbol {v}}_t'$

和

$\boldsymbol S_{10}=T^{-1}\sum_{t=1}^T\hat{\boldsymbol {u}}_t\hat{\boldsymbol {v}}_t'=\boldsymbol S_{01}$

这两个回归部分地消除了滞后差分$\triangledown \boldsymbol y_{t-1},...,\triangledown \boldsymbol y_{t-p+1}$的影响从$\triangledown \boldsymbol y_t$和$\triangledown \boldsymbol y_{t-1}$，使我们把注意力集中在理清这两个变量之间的关系，由$\boldsymbol \beta\boldsymbol \alpha'$参数化。向量$\boldsymbol \alpha$，通过估计$\boldsymbol y_{t-1}$的r个线性组合，其具有与$\triangledown \boldsymbol y_t$的最大平方偏相关：这被称为减秩回归。



15.9更精确地，此程序最大化（15.7）的似然，通过将其视为广义特征值问题并求解一组如下形式的方程：

$(\lambda_i\boldsymbol S_{11}-\boldsymbol S_{10}\boldsymbol S_{00}^{-1}\boldsymbol S_{01})\boldsymbol v_i=0,i=1,...,n$(15.8)

其中$\lambda_1\ge\lambda_2\ge...\ge\lambda_n\ge0$是特征值集，$\boldsymbol V=(\boldsymbol v_1,\boldsymbol v_2,...,\boldsymbol v_n)$包含相关的特征向量集，服从标准化

$\boldsymbol V'\boldsymbol S_{11}\boldsymbol V=\boldsymbol I_n$

$\boldsymbol \alpha$的ML估计由对应于r个最大特征值给出：

$\hat{\boldsymbol \alpha}=(\boldsymbol v_1,\boldsymbol v_2,...,\boldsymbol v_r)$

$\boldsymbol \beta$的ML估计值因此计算为$\hat{\boldsymbol \beta}=\boldsymbol S_{01}\hat{\boldsymbol \alpha}$，等价于将$\hat{\boldsymbol \alpha}$代入（15.7）得到的$\boldsymbol \beta$的OLS估计，还提供了模型中其余参数的ML估计。



15.10 当（15.7）中包括趋势，以及对截距和趋势系数施加各种限制时，程序可以直接适应。这涉及调整第一步和第二步回归来适应这些变化。再次考虑包括线性趋势的水平VAR（15.1）：

$\boldsymbol A(B)\boldsymbol y_t=\boldsymbol c+\boldsymbol dt+\boldsymbol u_t$(15.9)

通常，截距和趋势系数可以写成：

$\boldsymbol c=\boldsymbol \beta\boldsymbol \gamma_1+\boldsymbol \beta_{\perp}\boldsymbol \gamma_1^*,\boldsymbol d=\boldsymbol \beta\boldsymbol \gamma_2+\boldsymbol \beta_{\perp}\boldsymbol \gamma_2^*$

其中$\boldsymbol \beta_{\perp}$(垂直符号)是$n\times(n-r)$矩阵，称为$\boldsymbol \beta$的正交补，定义为使$\boldsymbol \beta_{\perp}'\boldsymbol \beta=\boldsymbol 0$，$\boldsymbol \gamma_1$和$\boldsymbol \gamma_2$为$r\times1$向量，$\boldsymbol \gamma_1^*$和$\boldsymbol \gamma_2^*$为$(n-r)\times1$向量。服从

$\boldsymbol \beta'\boldsymbol c=\boldsymbol \beta'\boldsymbol \beta\boldsymbol \gamma_1+\boldsymbol \beta'\boldsymbol \beta_{\perp}\boldsymbol \gamma_1^*=\boldsymbol \beta'\boldsymbol \beta\boldsymbol \gamma_1$

同样，$\boldsymbol \beta'\boldsymbol d=\boldsymbol \beta'\boldsymbol \beta\boldsymbol \gamma_2$。然后可以将关联的VECM写为

$\triangledown\boldsymbol y_t=\boldsymbol \Phi(B)\triangledown\boldsymbol y_{t-1}+\boldsymbol \beta_{\perp}(\boldsymbol \gamma_1^*+\boldsymbol \gamma_2^*t)+\boldsymbol \beta(\boldsymbol \gamma_1+\boldsymbol \gamma_2(t-1)+\boldsymbol e_{t-1})+\boldsymbol u_t$

如果$\boldsymbol \beta_{\perp}\boldsymbol \gamma_2^*=\boldsymbol 0$，即$\boldsymbol d=\boldsymbol \beta\boldsymbol \gamma_2$,趋势将被限制在误差校正。同样，如果$\boldsymbol \beta_{\perp}\boldsymbol \gamma_1^*=\boldsymbol 0$，截距将被限制在误差校正。因此，“包括趋势”的误差校正可以定义为$\boldsymbol e_t^*=\boldsymbol e_t+\boldsymbol \gamma_1+\boldsymbol \gamma_2t$。



15.11 当然，ML估计是基于已知的协整秩r，但实际上该值是未知的。幸运的是方程（15.8）还提供了一种确定r值的方法。如果r=n，且$\boldsymbol \Pi$是不受限制的，最大对数似然为

$\boldsymbol{\mathcal L} (n)=\boldsymbol K-(T/2)\sum_{i=1}^nlog(1-\lambda_i)$

其中

$\boldsymbol K=-(T/2)(n(1+2log2\pi)+log|\boldsymbol S_{00}|)$

对于给定的r\<n，只有前r个特征值应为正，并且限制的对数似然是

$\boldsymbol{\mathcal L} (r)=\boldsymbol K-(T/2)\sum_{i=1}^rlog(1-\lambda_i)$

LR检验假设存在r个协整向量的，备择假设为存在n个：

$n_r=2(\boldsymbol{\mathcal L} (n)-\boldsymbol{\mathcal L} (r))=-T\sum_{i=r+1}^nlog(1-\lambda_i)$

这称为迹统计量，检验序列$\eta_0,\eta_1,...,\eta_{n-1}$。如果最后一位显著的统计量是$\eta_{r-1}$，则选择r为协整秩，从而拒绝假设$\boldsymbol \Pi$有n-r+1个单元根。迹统计量衡量调整系数$\boldsymbol \beta$的重要性，在可能被忽略的特征向量上。



15.12另一种检验是评估最大特征值的显著性，用

$\zeta_r=-Tlog(1-\lambda_{r+1}),r=0,1,...,n-1$

称为最大特征值或λ最大统计量。$\eta_r$和$\zeta_r$都具有非标准极限分布，即Dickey-Fuller单位根分布的扩展。极限分布取决于n以及在VECM中对常数和趋势行为的限制。

这些检验通常称为Johansen系统协整检验：Johansen（1988，1995），首先提出并发展了这种方法。



## **向量误差校正模型的识别**

15.13 一个协整向量（r=1），施加一个限制足以识别协整向量。更一般地，假设$\boldsymbol \Pi$的秩为r，隐含加$(n-r)^2$个限制在其$n^2$个系数上，留下$n^2-(n-r)^2=2nr-r^2$个自由参数。两个$n\times r$矩阵，$\boldsymbol \alpha$和$\boldsymbol \beta$，包含2nr个参数，因此识别$\boldsymbol \Pi=\boldsymbol \beta\boldsymbol \alpha'$需要$r^2$个限制。

如果目前仅对$\boldsymbol \alpha$矩阵施加识别限制，如果它们是线性的，并且没有交叉协整向量限制，那么这些限制可以把第i个协整向量写为$\boldsymbol R_i\boldsymbol \alpha_i=\boldsymbol a_i$，其中$\boldsymbol R_i$和$\boldsymbol a_i$分别是$r\times n$矩阵和$r\times 1$向量。$\boldsymbol \alpha$唯一识别的必要和充分条件是每个$\boldsymbol R_i\boldsymbol \alpha_i$的秩为r，而必要条件是必须对r个协整向量中每个都施加r个限制。注意，仅通过对$\boldsymbol \alpha$本身的限制即可实现对$\boldsymbol \alpha$以及$\boldsymbol \Pi$的识别。无法通过短期动态限制识别长期关系：因此，（15.6）的$\boldsymbol \Phi_i$系数可以自由估计。



15.14如果对$\boldsymbol \alpha$施加的限制数为k，则将k设置为$r^2$构成精确识别。对r个协整向量中每一个强加r个限制不会改变似然$\boldsymbol{\mathcal L} (r)$，因此，虽然它们的强加使得可以获得唯一的$\boldsymbol \alpha$估计，但是限制的有效性无法检验。通常，r个限制通过归一化获得，如果r=1，那么这就是所需要的。对于r\>1，进一步需要$r^2-r$个限制（每个方程$r-1$个），这构成了Phillips（1991）三角形表示的基础。将$\boldsymbol \alpha$写为

 $\boldsymbol \alpha'=[\boldsymbol I_r \quad -\boldsymbol \Gamma]$

其中$\boldsymbol \Gamma$是$r\times (n-r)$矩阵。因此$r^2$个刚刚识别的限制，由r个归一化和$r^2-r$个零限制组成，对应解$\boldsymbol \alpha'\boldsymbol y_t$,对$\boldsymbol y_t$的前r个分量。



15.15 当$k>r^2$，有$k-r^2$个过度识别限制。如果$\boldsymbol{\mathcal L} (r:q)$表示施加$q=k-r^2$个过度识别限制后的对数似然，则可以使用LR统计量$2(\boldsymbol{\mathcal L} (r)-\boldsymbol{\mathcal L} (r:q))$，渐近分布为$\chi^2(q)$。



15.16 也可以对$\boldsymbol \beta$施加限制，并且可以将$\boldsymbol \alpha$和$\boldsymbol \beta$联系起来。限制的来源是考虑一些变量弱外生性的假设。假设我们分割$\boldsymbol y_t=(\boldsymbol x_t',\boldsymbol z_t')'$，其中$\boldsymbol x_t$和$\boldsymbol z_t$是$n_1\times 1$和$n_2\times 1$向量，$n_1+n_2=n$，VECM（15.7）写为一对“边际”模型

$\triangledown\boldsymbol x_t=\boldsymbol c_1+\sum_{i=1}^{p-1}\boldsymbol \Phi_{1,i}\triangledown\boldsymbol y_{t-i}+\boldsymbol \beta_1\boldsymbol \alpha'\boldsymbol y_{t-1}+\boldsymbol u_{1,t}$(15.10a)

$\triangledown\boldsymbol z_t=\boldsymbol c_2+\sum_{i=1}^{p-1}\boldsymbol \Phi_{2,i}\triangledown\boldsymbol y_{t-i}+\boldsymbol \beta_2\boldsymbol \alpha'\boldsymbol y_{t-1}+\boldsymbol u_{2,t}$(15.10b)

其中

$\boldsymbol \Phi_i=\left[\begin{array}{}\boldsymbol \Phi_{1,i}\\\boldsymbol \Phi_{2,i}\end{array}\right],i=1,...,m-1,\boldsymbol \beta=\left[\begin{array}{}\boldsymbol \beta_1\\\boldsymbol \beta_2\end{array}\right],\boldsymbol u_t=\left[\begin{array}{}\boldsymbol u_{1,i}\\\boldsymbol u_{2,i}\end{array}\right]$

是一致的分割。$\boldsymbol z_t$弱外生对$(\boldsymbol \alpha,\boldsymbol \beta_1)$的条件为$\boldsymbol \beta_2=\boldsymbol 0$，在这种情况下，误差校正$\boldsymbol e_t=\boldsymbol \alpha'\boldsymbol y_t$不进入$\boldsymbol z_t$的边际模型。

可以通过包含对$\boldsymbol \beta$加$n_2$个零限制来检验这种弱外生性假设，也就是在零假设下，$\boldsymbol \beta=[\boldsymbol \beta_1 \quad \boldsymbol 0]'$，15.15中概述的LR检验加q个过度识别限制。



## 结构向量误差校正模型

15.17 继Johansen和Juselius（1994）之后，“结构性VECM”可能被写成

$\boldsymbol \Gamma_0\triangledown\boldsymbol y_t=\sum_{i=1}^{p-1}\boldsymbol \Gamma_i\triangledown\boldsymbol y_{t-i}+\boldsymbol \Theta\boldsymbol \alpha'\boldsymbol y_{t-1}+\boldsymbol v_t$(15.11)

与“简化型” VECM有关

$\triangledown\boldsymbol y_t=\sum_{i=1}^{p-1}\boldsymbol \Phi_i\triangledown\boldsymbol y_{t-i}+\boldsymbol \beta\boldsymbol \alpha'\boldsymbol y_{t-1}+\boldsymbol u_t$

通过

$\boldsymbol \Gamma_i=\boldsymbol \Gamma_0\boldsymbol \Phi_i,i=1,...,p-1$

$\boldsymbol \Gamma_0\boldsymbol \beta=\boldsymbol \Theta,\boldsymbol v_t=\boldsymbol \Gamma_0\boldsymbol u_t$

因此

$E(\boldsymbol v_t\boldsymbol v_t')=\boldsymbol \Gamma_0\boldsymbol \Omega_p\boldsymbol \Gamma_0'$

注意，此框架假设协整向量已经被识别（及其参数集），以便识别“短期”结构，即参数集$\boldsymbol \Gamma_0,\boldsymbol \Gamma_1,...,\boldsymbol \Gamma_{p-1},\boldsymbol \Theta$,有条件地以$\boldsymbol \alpha$的形式进行。这可以通过使用传统方法，通常会以探索性方式进行，很少会了解关于短期结构的先验知识。



## **向量误差校正模型中的因果关系检验**

15.18考虑边际VECM（15.10a，b）的“完全分割”形式：

$\triangledown\boldsymbol x_t=\boldsymbol c_1+\sum_{i=1}^{p-1}\boldsymbol \Phi_{11,i}\triangledown\boldsymbol x_{t-i}+\sum_{i=1}^{p-1}\boldsymbol \Phi_{12,i}\triangledown\boldsymbol z_{t-i}+\boldsymbol \beta_1\boldsymbol \alpha_2'\boldsymbol z_{t-1}+\boldsymbol u_{1,t}$

$\triangledown\boldsymbol z_t=\boldsymbol c_2+\sum_{i=1}^{p-1}\boldsymbol \Phi_{21,i}\triangledown\boldsymbol x_{t-i}+\sum_{i=1}^{p-1}\boldsymbol \Phi_{22,i}\triangledown\boldsymbol z_{t-i}+\boldsymbol \beta_2\boldsymbol \alpha_2'\boldsymbol z_{t-1}+\boldsymbol u_{1,t}$

其中现在

$\boldsymbol \Phi_i=\left[\begin{array}{}\boldsymbol \Phi_{11,i}&\boldsymbol \Phi_{12,i}\\\boldsymbol \Phi_{21,i} &\boldsymbol \Phi_{22,i}\end{array}\right],\boldsymbol \alpha'=[\boldsymbol \alpha_1\quad\boldsymbol \alpha_2]'$

$\boldsymbol z$不是$\boldsymbol x$的Granger-因的假设可以形式化为

$\boldsymbol{\mathcal H}_0:\boldsymbol\Phi_{12,1}=...=\boldsymbol\Phi_{12,p-1}=\boldsymbol0,\boldsymbol\beta_1\boldsymbol\alpha_2'=\boldsymbol0$

$\boldsymbol{\mathcal H}_0$的第二部分，通常称为“长期非因果关系”，牵涉到$\boldsymbol \alpha$和$\boldsymbol \beta$系数的非线性函数，使检验相当复杂：参见Toda和Phillips（1993，1994）。基于非受限$\boldsymbol \Pi$矩阵的检验，即，$\boldsymbol \Pi_{12}=\boldsymbol 0$，使用一个明显的记号，是无效的，因为Wald检验统计量分布只渐近为$\chi^2$如果已知$\boldsymbol \alpha_2$秩为$n_2$，这是估计“水平”VAR不提供的信息。



15.19由于检验$\boldsymbol{\mathcal H}_0$的复杂性，一个更简单，但不可避免地功效较弱且效率较低的程序，由Toda和Yamamoto（1995）和Saikkonen和Lütkepohl（1996）提出。假设我们在水平上考虑VAR(p)，但现在将阶增加一，即我们拟合VAR(p+1)。事实证明，现在可以检验非因果假设，由传统的Wald统计量得出，因为通过假设$\boldsymbol \Phi_{12,p+1}=\boldsymbol 0$额外的滞后，允许再次使用标准渐近推断。

如果VAR中的变量数量很少并且滞后阶数相当大，那么包括额外的滞后可能只会导致较小的效率低下，因此，考虑到可以轻松构建检验，在这种情况下,“滞后增大”VAR（LA-VAR）方法应予以认真考虑。



## **非平稳VARs的脉冲响应渐近性**

15.20 VAR各种脉冲响应是根据矩阵序列计算的

$\boldsymbol \Psi_i=\sum_{j=1}^i\boldsymbol A_j\boldsymbol \Psi_{i-j},\boldsymbol \Psi_0=\boldsymbol I_n,\boldsymbol \Psi_i=\boldsymbol 0,i<0$

在非平稳VAR中，其计算仍然完全相同，但是如果$\boldsymbol \Pi=-\sum_{j=1}^p\boldsymbol A_j$是降秩的，$\boldsymbol \Psi$的元素不会随着i增加消亡，浙江导致一些分析上的复杂性。



15.21在平稳的VAR，其中长期矩阵$\boldsymbol \Pi$的所有根小于1，估计的脉冲响应可能显示为一致，渐近正态，$\boldsymbol \Psi_i$及其估计$\hat{\boldsymbol \Psi}_i$都趋于零。对于非平稳的VAR，其中$\boldsymbol \Psi_i$不一定消失，随着$i\to\infty$，对脉冲响应的估计采用了另一种极限理论，如Phillips（1998）和Stock（1996）的结果显示。

这些结果总结在这里。如果系统中存在单位根，根据一个水平VAR的OLS估计的长范围（大i）脉冲响应不一致；估计响应的极限值是随机变量，而不是真正的脉冲响应。原因是，这些真正的脉冲响应不会随着i增加消亡，它们会无限期地带有单位根的影响。既然单位根的估计带有误差（即，估计的根不完全趋于1），估计误差的影响会在极限范围内持续，随着$T\to\infty$。$\hat{\boldsymbol \Psi}_i$的极限分布是不对称的，脉冲响应的置信区间也将是不对称的。

另一方面，CVAR中的极限脉冲响应一致估计，如果协整的秩是已知的或本身估计一致，通过15.10‑15.11的检验或使用信息标准。这是因为在降秩回归中，估计矩阵乘积$\boldsymbol \beta\boldsymbol \alpha'$而不是$\boldsymbol \Pi$，所以没有单位根被估计（隐式或显式）。尽管如此，这些一致的秩选择程序往往会错误地取接近1的根，而实际为1，因此，脉冲响应估计将收敛到非零常数而并非消亡，并伴随相当宽的置信区间。

因此，对于非平稳VAR的脉冲响应不应该由不受限制的水平VAR计算。既然知道系统的单位根数对于获得准确的估算是必要的，确保通过实践中很好的一致方法选择协整秩很重要。



## **向量误差校正X型模型**

15.22  CVAR / VECM模型的直接扩展是包括$I(0)$外生变量的向量，$\boldsymbol w_t$，其可以输入每个方程：

$\triangledown\boldsymbol y_t=\boldsymbol c+\boldsymbol dt+\sum_{i=1}^{p-1}\boldsymbol \Phi_i\triangledown\boldsymbol y_{t-i}+\boldsymbol \beta\boldsymbol \alpha'\boldsymbol y_{t-1}+\boldsymbol \Lambda\boldsymbol w_t+\boldsymbol u_t$(15.12)

协整秩的估计和检验仍然与以前一样，尽管检验的临界值可能会受到影响。



## **共同趋势和周期**

15.23 CVAR中存在线性趋势进一步含义的最佳分析是通过引入无序向量多项式$\boldsymbol C(B)$，定义为

$\boldsymbol C(B)\boldsymbol \Pi(B)=\triangledown\boldsymbol I_n$(15.13)

类似于15.1中$\boldsymbol A(B)$的分解，我们有

$\boldsymbol C(B)=\boldsymbol I_n+\boldsymbol CB+(\boldsymbol C_1^*B+\boldsymbol C_2^*B^2+...)\triangledown=\boldsymbol I_n+\boldsymbol C+(\boldsymbol C_0^*+\boldsymbol C_1^*B+\boldsymbol C_2^*B^2+...)\triangledown \\=\boldsymbol I_n+\boldsymbol C+\boldsymbol C^*(B)\triangledown=\boldsymbol C(1)+\boldsymbol C^*(B)\triangledown$

$\boldsymbol C(B)$的矩阵$\boldsymbol C_0,\boldsymbol C_0,...,$由递归给出

$\boldsymbol C_i=\sum_{j=1}^p\boldsymbol C_{i-j}\boldsymbol A_j,i>0,\boldsymbol C_0=\boldsymbol I_n$

因此

$\boldsymbol C=\sum_{i=1}^\infty\boldsymbol C_{i}=\boldsymbol C(1)-\boldsymbol I_n$

有

$\boldsymbol C_0^*=-\boldsymbol C$

和

$\boldsymbol C_i^*=\boldsymbol C_{i-1}^*+\boldsymbol C_i,i>0$

VAR（15.9）可以写为

$\triangledown\boldsymbol y_t=\boldsymbol C(B)(\boldsymbol c+\boldsymbol dt+\boldsymbol u_t)=(\boldsymbol C(1)+\boldsymbol C^*(B)\triangledown)(\boldsymbol c+\boldsymbol dt)+\boldsymbol C(B)\boldsymbol u_t \\=\boldsymbol C(1)\boldsymbol c+\boldsymbol C^*(1)\boldsymbol d+\boldsymbol C(1)\boldsymbol dt+\boldsymbol C(B)\boldsymbol u_t=\boldsymbol b_0+\boldsymbol b_1t+\boldsymbol C(B)\boldsymbol u_t$

其中

$\boldsymbol b_0=\boldsymbol C(1)\boldsymbol c,\boldsymbol b_1=\boldsymbol C(1)\boldsymbol d$

在水平，变为

$\boldsymbol y_t=\boldsymbol y_0+\boldsymbol b_0t+\boldsymbol b_1\frac{t(t+1)}{2}+\boldsymbol C(B)\sum_{s=1}^t\boldsymbol u_s=\boldsymbol y_0+\boldsymbol b_0t+\boldsymbol b_1\frac{t(t+1)}{2}+(\boldsymbol C(1)+\boldsymbol C^*(B)\triangledown)\sum_{s=1}^t\boldsymbol u_s\\=\boldsymbol y_0+\boldsymbol b_0t+\boldsymbol b_1\frac{t(t+1)}{2}+\boldsymbol C(1)\boldsymbol s_t+\boldsymbol C^*(B)(\boldsymbol u_t-\boldsymbol u_0)=\boldsymbol y_0^*+\boldsymbol b_0t+\boldsymbol b_1\frac{t(t+1)}{2}+\boldsymbol C(1)\boldsymbol s_t+\boldsymbol C^*(B)\boldsymbol u_t$(15.4)

其中

$\boldsymbol y_0^*=\boldsymbol y_0-\boldsymbol C^*(B)\boldsymbol u_0,\boldsymbol s_t=\sum_{s=1}^t\boldsymbol u_s$



15.24 因此，VAR（15.9）中包含线性趋势，$\boldsymbol y_t\sim I(1)$，因此表示水平方程（15.14）有二次趋势。此外，由于$\boldsymbol b_1=\boldsymbol C(1)\boldsymbol d$，只有$\boldsymbol C(1)=\boldsymbol 0$时，这种二次趋势才会消失。从（15.13），$\boldsymbol C(1)\boldsymbol A(1)=\boldsymbol 0$，因此$\boldsymbol C(1)=\boldsymbol 0$需要$\boldsymbol A(1)=-\boldsymbol \Pi\ne\boldsymbol 0$。仅在$\boldsymbol A(B)$不包含因子1-B的情况下，即$\boldsymbol y_t\sim I(0)$，这已被假设排除，但暗示$\boldsymbol \Pi$是满秩n。

然而如果，$\boldsymbol A(1)=\boldsymbol 0$，因此$\boldsymbol \Pi=\boldsymbol 0$，为零秩且含有n个单位根，则没有协整和$\boldsymbol C(1)$，因此$\boldsymbol b_1$是无约束的。在一般情况下，其中$\boldsymbol \Pi$的秩为r，则遵循$\boldsymbol C(1)$的秩为n-r。$\boldsymbol b_1$的秩，因此独立二次确定趋势的数量，也等于n-r，并且将随着协整秩r的增加而减小。



15.25在不限制趋势系数$\boldsymbol b_1$的情况下，（15.14）的解将具有以下性质：$\boldsymbol y_t$中趋势的本质将随协整向量的数量变化。为了避免这种令人不满意的结果，可以施加限制$\boldsymbol b_1=\boldsymbol C(1)\boldsymbol d=\boldsymbol 0$，在这种情况下$\boldsymbol y_t$的解将仅包含线性趋势,不考虑r的值。r的选择确定独立线性确定趋势数r和模型中的随机趋势数n-r之间的分离。



15.26然后考虑（15.14）,施加限制$\boldsymbol b_1=\boldsymbol 0$，并且简单地，初始值$\boldsymbol y_0=\boldsymbol u_0=\boldsymbol 0$：

$\boldsymbol y_t=\boldsymbol b_0+\boldsymbol C(1)\boldsymbol s_t+\boldsymbol C^*(B)\boldsymbol u_t=\boldsymbol C(1)(\boldsymbol c+\boldsymbol s_t)+\boldsymbol C^*(B)\boldsymbol u_t$(15.15)

如果存在协整，那么正如我们已经看到的，$\boldsymbol C(1)$是降秩的h=n-r且可以写为乘积$\boldsymbol \rho\boldsymbol \delta'$，其中两个矩阵都是秩h。定义

$\boldsymbol \tau_t=\boldsymbol \delta'(\boldsymbol c+\boldsymbol s_t),\boldsymbol c_t=\boldsymbol C^*(B)\boldsymbol u_t$

然后（15.15）可以用Stock和 Watson（1988）的“共同趋势”表示：

$\boldsymbol y_t=\boldsymbol \rho\boldsymbol \tau_t+\boldsymbol c_t$

$\boldsymbol \tau_t=\boldsymbol \tau_{t-1}+\boldsymbol \delta'\boldsymbol u_t$(15.16)

该表示将$\boldsymbol y_t$表示为h=n-r个随机游走的线性组合，这是共同趋势$\boldsymbol \tau_t$，再加上一些平稳的“瞬时”分量$\boldsymbol c_t$。实际上，（15.16）可以看作是Beveridge-Nelson分解的多元扩展。通过与关于15.7中的协整矩阵$\boldsymbol \alpha$的论点相似，$\boldsymbol \delta$没有唯一定义，因此没有引入一些其他的识别限制（参见Wickens，1996）。



15.27 以同样的方式，共同趋势出现在$\boldsymbol y_t$中，当$\boldsymbol C(1)$是减秩的，如果$\boldsymbol C^*(B)$是减秩的，则出现共同周期，既然$\boldsymbol c_t=\boldsymbol C^*(B)\boldsymbol u_t$是$\boldsymbol y_t$的周期性分量。共同周期的存在要求有$\boldsymbol y_t$元素的线性组合不包含以下周期性分量：也就是说，存在一组s个线性独立的向量，聚集在$n\times s$矩阵$\phi$中，使得

$\phi'\boldsymbol c_t=\phi'\boldsymbol C^*(B)\boldsymbol u_t=\boldsymbol 0$

在这种情况下

$\phi'\boldsymbol y_t=\phi'\boldsymbol \rho\boldsymbol \tau_t$

这样的矩阵存在，如果所有的$\boldsymbol C^*_i$秩少于满秩，且如果$\phi'\boldsymbol C^*_i=\boldsymbol0$，对所有的i; 这是Vahid和Engle（1993）得出的结果。在这些情况下，我们可以写$\boldsymbol C^*_i=\boldsymbol G \boldsymbol{\widetilde C}_i$，对于所有i，其中$\boldsymbol G$是$n\times(n-s)$矩阵，具有满秩，且$\boldsymbol{\widetilde C}_i$可能不具有满秩。定义$\boldsymbol{\widetilde C}(B)=\boldsymbol{\widetilde C}_0+\boldsymbol{\widetilde C}_1B+...$,周期分量可写为

$\boldsymbol c_t=\boldsymbol G\boldsymbol{\widetilde C}(B)\boldsymbol u_t=\boldsymbol G\boldsymbol{\widetilde c}_t$

这样n元素周期$\boldsymbol c_t$可以写成(n-s)元素周期$\boldsymbol{\widetilde c}_t$的线性组合，从而导出共同趋势共同周期表示

$\boldsymbol y_t=\boldsymbol \rho\boldsymbol \tau_t+\boldsymbol G\boldsymbol{\widetilde c}_t$

组成$\phi$的线性独立“共同特征”向量的数量s最多为h=n-r，它们与组成$\boldsymbol\alpha$的协整向量线性独立。这是由于共同趋势的向量$\phi'\boldsymbol y_t$，是$I(1)$，而误差校正向量$\boldsymbol\alpha'\boldsymbol y_t$，是$I(0)$。



15.28 当r+s= n，（15.17）表示的一个有趣的特例发生,因为在这种情况下$\boldsymbol y_t$具有唯一的趋势周期分解$\boldsymbol y_t=\boldsymbol y_t^\tau+\boldsymbol y_t^c$，其中

$\boldsymbol y_t^\tau=\boldsymbol \Theta_1\phi'\boldsymbol y_t=\boldsymbol \Theta_1\phi'\boldsymbol \rho\boldsymbol \tau_t$

包含随机趋势且

$\boldsymbol y_t^c=\boldsymbol \Theta_2\boldsymbol\alpha'\boldsymbol y_t=\boldsymbol \Theta_2\boldsymbol\alpha'\boldsymbol c_t$

包含周期性分量。这里

$[\boldsymbol \Theta_1\quad \boldsymbol \Theta_2]=\left[\begin{array}{}\boldsymbol \alpha'\\\phi'\end{array}\right]^{-1}$

注意$\boldsymbol y_t^c$为误差校正$\boldsymbol e_t=\boldsymbol\alpha'\boldsymbol y_t$的线性组合。既然$\boldsymbol y_t^\tau$和$\boldsymbol y_t^c$是$\boldsymbol \alpha$和$\phi$的函数，可以很容易地计算出作为$\boldsymbol y_t$的简单线性组合。



15.29  只有在可逆变换之前，才能识别$n\times s$共特征矩阵$\phi$，因为$\phi$列的任何线性组合也将是一个共特征向量。因此，矩阵可以旋转为具有s维识别子矩阵

$\phi=\left[\begin{array}{}\boldsymbol I_s\\\phi^*_{(n-s)\times s}\end{array}\right]$

使用此规范，可以将s个共特征向量以及r个协整向量并入一个VECM，其中考虑$\phi'\triangledown\boldsymbol y_t$作为$\triangledown\boldsymbol y_t$前s个元素的s“伪结构形式”方程。然后通过对$\triangledown\boldsymbol y_t$其余的n-s个方程添加无约束的VECM方程来完成系统，得到：

$ \left[\begin{array}{}\boldsymbol I_s &\phi^{*'} \\ \boldsymbol 0_{(n-s)\times s} & \boldsymbol I_{n-s}\end{array}\right]\triangledown\boldsymbol y_t=\left[\begin{array}{}\boldsymbol 0_{s\times (n(p-1)+r)}\\ \boldsymbol\phi_1^*,...,\boldsymbol\phi_{p-1}^*\boldsymbol\beta^*\end{array}\right]\left[\begin{array}{}\triangledown\boldsymbol y_{t-1}\\\vdots\\\triangledown\boldsymbol y_{t-p+1}\\\boldsymbol e_{t-1}\end{array}\right]+\boldsymbol u_t$(15.18)

其中$\boldsymbol\phi_1^*$包含等$\boldsymbol\phi_1$的最后n-s行。s共同周期的存在因此意味着$\phi'\triangledown\boldsymbol y_t$独立于$\triangledown\boldsymbol y_{t-1},...,\triangledown\boldsymbol y_{t-p+1}$和$\boldsymbol e_{t-1}$，因此$\boldsymbol y_t$的所有过去值。



15.30  可以通过全信息最大似然或其他联立方程估计技术来估计系统（15.18）。可以构造施加s个共特征向量限制的似然比统计量，渐近分布为$\chi^2$，自由度由已经被限制的数量决定。

忽略截距，VECM(15.6)有n(n(p-1)+r)个参数，而伪结构模型（15.18）在前s个方程中有$sn-s^2$个参数，在完成系统的n-s个方程中具有$(n-s)(n(p-1)+r)$个参数，因此总共施加了$s^2+sn(p-1)+sr-sn$个限制。注意，如果p=1和r=n-s，则限制数为零。系统刚刚识别，无需检验共同周期，因为系统必定会有r个共同周期。随着滞后阶数p增加，系统通常会被过度识别，共同周期的检验变得必要。
