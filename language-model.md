# Language Model 語言模型

根據前面的 word 讓機器知道後面的 word 應該選什麼比較合理，使聲音相似的一句話「e.g. 我癌膩」應該是「我愛你」這樣才是合語意的。





### Perplexity

H = 8，可以想成2^8 = 256 個字，這是「虛擬的詞典的大小」，是 2^H 次方，是混淆的程度，稱為 perplexity。舉例是 uniform distri. 。 unigram 算出來的數字。

### Entropy

看公式，每個 word 取 log 然後再乘上自己的 prob，類似期望值，就是 entropy。以 unigram 舉例，（i.e. 目前只讀入第一個字，沒有前面字的資訊）就是把 prob 帶入 1/1024 。



### Uncertainty: 選詞的難度

雖然機率高高低低，但因為 entropy 都是一樣的（都是 H = 10 bits），所以 uniform distri 和正常的 distri 他們的 uncertainty 一樣。



### Branching Factor Estimate

每次要做多少選擇？相當於要做 1024 次選擇，有這麼多種分岔，然後 random 選一個分岔走下去。

**引入 language model 可以大幅降低。**



## 舉例

假設 unigram, bigram, 然後後面都是 trigram，他們的機率都已經算好給你了（ 1/1024, 1/512, 1/256… ）

把 n 個乘起來再開 n 次方根，就是倒數後、再來做幾何平均。混淆度。

### averging branching factor

不管哪種 gram（就是 n-gram 的概念），算出平均任何一個字選對的機率有多難。

平均我每選一個詞，就是和在 312 個字之中任選一個的難度差不多。



### 都是給定前面 1, 2, 3 or more 個字，算他的機率



$C_i$  就是給定前面 i 個 word ，算條件機率



### Test Corpus 語料庫 $D$

測試 language model 的 dataset，$N$ 是 10000，就是有 10000 句， $W_i$ 就是第 i 個句子，每一句有 $n_i$ 個詞，有就是有小 w： $w_{i1}, w_{i2}…w_{in}$ 這麼多個詞，共 $n_i$ 個詞。然後 $N_D$ 就是加總總共的詞數量，$n_i$ 全部 sum up。



### pp : perplexity

這個 language model 在這個 dataset 上面的 constriant 表現是 PP = 312 ，也就是平均我每選一個詞，就是和在 312 個字之中任選一個的難度差不多。算法就是把他每個 prob 乘起來、然後取平均、這樣是算出 entropy。然後再把 entropy 放在 2 的指數次方，算出來就是 perplexity = 312。越低越好，代表我的 language constraint 越強、代表和在 「越少」的字之中任選一個的難度差不多。 e.g. 體育新聞，用的詞彙有限、句型單純，結果即使 training data 很少還是能表現很好，pp 很小。e.g. 文化教育新聞，就用很多很多詞、 句型複雜、pp 就很高。



pp: a function of test dataset $D$ (corpus) and language modeling $P(w_i|c_i)$ 。

要算這個是要用 test set 算出來。就是 在 testing set 的表現好壞的概念。



## Domain-specific model



1. Acoustic Model: 最好是每個人一個，因為大家聲音不同
2. Language Model: 最好是每個 topic 一個，因為用的專有名詞、句子形狀差不多。



## Perplexity 是一種 cross-entropy



### 複習： KL Divergence

$D[p(x) || q(x)] = \sum p(x_i)log\frac{p(x_i)}{q(x_i)} \geq 0$

p, q 是兩個分佈，他們的 KL divergence 是這樣算出來的。想成一個 distance。

把他移向過去就是 *Jensen's Inequality* ，如下：



$-\sum p(x_i)log(p(x_i)) \leq -\sum p(x_i)log(q(x_i))$

這句是啥意思？

右半邊：

Cross-entropy 通常被叫做 $p * log(q)$， q 跟 p 差越多的話就會值越大

左半邊：

log 裡面的 p 跟 外面的 p 是同一個分佈的話，乘起來 sum up 就是下限，也就是 entropy

### 為什麼 perplexity 和右半邊是同樣的，都是 cross-entropy ？？

根據大數法則。

大數法則：丟骰子，丟超多次就會幾乎是個「固定的機率」，在大數之下每次出來都是 prob 這麼多的可能性分啊聲。

q 就是我的 n-gram，我在 training set 訓練的、試圖去逼近 target function $f$ 的那個 $h$。

p 就是每一種 n-gram 在 testing corpus 出現的真實機率，就是 $f$。

training corpus 和 testing corpus 的 n-gram 分佈越接近、最後出來 pp 就會越好（越小）



$$\lim_{N\to\infty} \frac{1}{N} log (q(x_k))$$ 因為 N 太大，所以看起來就像 logq 乘上 機率，就會趨近於 $\sum p(x_i) log(q(x_i))$  相加。



