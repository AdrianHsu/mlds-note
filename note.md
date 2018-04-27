# 0309

network structure比較好 -> 參數多x

given function f(x) -> unit 數目 （neuron 數目） != parameter 數目

## 為什麼 deep 會比較 fit 得好？

reLU, 少用 sigmoid 了

e.g. `y = x^2` 

shallow network cover 的 function space 比較小，可能找不到一組參數能 fit target function

這課只討論能不能 fit ，不討論 optimization 

i.e. 
假設 function space 一定包含 target function & 假設若 function space 包含 target function 則一定可以找到 target function

# Shallow network 如何 fit 任何 conti. function

本圖 output layer 只做 weighted sum 沒有放 act. function。

piece-wise linear function
由一個個線段所構成

## L-Lipschitz func 是啥

比較 smooth 的 function

定義式：

L 是常數

output 變化會被 bound 在一個範圍內，被 L 跟 input 給 bound 住

output 的變化一定 <= input 變化

藍色線不是、而綠色是 L-lips

shallow network := 只有一個 hidden layer 

approximate? 

找到某個 K 讓 f 和 f* 很像

最大差距 綠色虛線 <= given \\( \epsilon\\)

用右上角的 L2 norm 也可以算，不一定要用 max norm
 假設 max 那條式子被滿足，則下面的 sqrt 那條 L2 norm 也會被滿足了。 proof?

0~1 之間的積分 <= epsilon

面積、高。如果 高 <= epsilon ，因為底是在 0~1 所以面積也會 <= epsilon 得證

所以只討論 max 那條的 case 就好


希望 piecewise line function 和 target func 差距 <= epsilon

怎算？在 f* 上點一些綠點，

l  兩點之間距離

L 是 L-lips 的 L 常數

一定 error <= l x L， why?

X 軸是 input， Y 是 output ，看 x 上的 l  距離只要比 y 的 L 小就好ㄋ


L 是給定的，因為f* 給定。但 l 可以操控，讓 l 越來越小則 error 也會變小，直到 error < epsilon 就好了

error: 要求要多緊貼的 fit

假設每個寬都是 epsilon / L

有沒有辦法製造 piecewise line function 那條綠線， 利用 shallow net?


## 用兩個 relu neurons 就可以輕鬆製造藍色的某條線，製造一堆藍色 function 再把它疊起來就可以製造出綠色線（piecewise lin. func）


ramp function 就是長得像藍線那樣

拿ramp func 1. 和 2. 加在一起，就可以得到

given L-lips 如何 approx 這條綠線？

需要 L/e 個 segments 也就是需要 2L/e 個 relu neur. 

# 用 deep 的理由？

neurons 數量到底要多少？

如果是shallow  的話需要 2L/epsilon 個


SVM 加上 kernel 是如何實作的？

x1 到 xn 每一筆 data 都計算相似度，不同kernal 代表不同相似度指標，很像matching 的兩行「萬能」演算法。

比方：疊在一起的 gate 能用很少個就做出 parity check，多個 gates 不疊的話會需要很多個才能達到一樣 function


reLU:
input=output, 
output=0,
是 piecewise linear func

activation pattern 某一種 mode 的組合e.g. lin, lin, 0, 0, lin, lin

國小數學，共有 2^n 個 activs pattern （pieces），這是 upper bound

但事實上，假設 n 個，其實只是 O(n) 沒辦法弄出 2^n 種 pieces


## 取絕對值的 Abs Activa. function

把兩個 reLU 組在一起，

if wx + b > 0 拿上路的結果，  < 0 則拿下路的結果

折線圖，推算說下一個 neuron 跟 最一開始的 x 的關係是啥

為什麼綠虛線是 1/2?

當我們用 deep structure 每次多加上兩個 neuron 後，則 neu regions 的線段數目就會 * 2 倍

100 個線段，則只需要 7 個 neurons 因為 2^7 > 100 就可以，就像雪花結晶一直 2^n 產生

### If K is width, H is depth, we can have at least K^H pieces(線段）

experiment result

pieces 類似雪花的對稱性， v 形狀變成 w 形狀，再變成 ww 形狀...就像摺紙一樣

### low-layer 的參數是比較重要的

## f(x) = x^2


如果要用 2^m 個依樣寬的片段去 fit f(x) 那 2^m 一定要 >= 1/2 (1/sqrt(e)) pieces，這結果比 2L/e 小

p.32

上圖減掉中圖 就是藍線
中圖減掉下圖 就是綠線

## Best of Shallow: 假如要 fit y=x^2 需要多少 pieces?

故意不讓黑線跟紅色的某兩點頭尾相接，而是直接疊上去，這樣 max error 一定會變小
這次只看平均 sqrt 的積分結果（euclidean norm），不要看 max 的了。給他夢幻狀態看 shallow 能不能贏 deep


Holder's Inequality

l 一定是正的因為是線段的長度

結論： shallow 的最佳狀態還是贏不過 deep，才 O(1/sqrt(e)) 而已

deep fit function
exponen 的好， y=x^2 要fit用 deep 會比較好，只要比 y=x^2 複雜都會很fit

# 0316

* Regression: 用 MSE
* Classification: 用 Cross-entropy
* 為什麼用 GD 就能在 non-convex 狀況就得到不錯的 solution？

## Optimization != learning

* 只問 loss func 給定，能不能找到 theta*？ 
* 會不會 overfit 暫時不管

## Loss 

* 有很多 global minimum ，而值都一樣
* 例如：交換兩個同一層的 neurons ，最後 loss 不變、但參數會改變，有點像是某個矩陣的某行換到另一行之類的而已。

## 走到某個 critical point

* critical point 的 gradient = 0
* 有可能不是 local minima
* eigen value 怎麼找？

## Degenerate
往某個方向走不增也不降
* H 裡面有 zero vector 的話，就看其他小項是正還負，這時候就不能用 Hessian 去猜他是 local max/min or saddle pt 了。
* 算出 H 是都是 0但是很明顯有 local max，只有一個大平台。但看更大維度的話其實是有變化的
* Monkey Saddle: 多的一個凹處給猴子的尾巴擺放。一般的saddle 只有兩個凹處
* 根本沒走到 saddle pt，即使已經 training loss平緩了
* 變小在暴增、變小在暴增...（進到 saddle pt 所以下降，跨出 saddle pt 所以增大）

* 先算 hassian 看他的 eigen value 是否都正/負 判斷是不是 global max/min
* hassian matrix = 0, 不管走去哪都是零 so flat!


# 逃出 saddle points 的 demo
* 兩層： std = 0.0, v.s. std = 0.00000001
* 三層： std = 0.0000001 仍然出不去，因為 hassian 都是 0 平坦區
* 如果 std = 1 就可以降 loss 了，或是 random normal(0, 1) 也可以

saddle pt 沒有 negative eigen value 會不能降 loss，在一層不會發生，兩層以上就有可能。

* deep linear network 沒有 local minima!
* linear network: 沒有加 activ. function layer ，全部都是 W(W(Wx + b) + b) + b)... 矩陣相乘而已




