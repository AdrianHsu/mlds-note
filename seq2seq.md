# Seq2seq

#### 什麼是 peephole?

#### LSTM

*  四個 linear transform （四色的粗箭頭）
* 假設 diagonal，$h^{t-1}$ 做了 transform 變成 $c^t$ 然後再變成 $h^t$ 
* 分成 zf, zi, z, zo



#### GRU?

h 比較像是 LSTM 裡面 c 做的事

1. 把h, x 接起來
2. 乘上 transform 得到 r (reset gate) 以及另一個 z (update gate)
3. 粗箭頭代表 transform，細箭頭就只是操作
4. 得到 $h'$ 然後 和 $1-z$ 做 elementwise 相乘
5. forget, input gate 兩個連動的 LSTM，就是一個 GRU
6. $h^t$ 乘上灰色的 transform 就是 $y^t$



### Generation：生成的過程



1. 產生第一個 char? 給機器特殊字元，只有在句子開頭（BOS, begin of sentence），對應第一維
2. y1 = 給定 BOS 算條件機率
3. 用 sample 說出「床」，表示成 vector。如果用 argmax 那就每次都一樣了
4. y2 是機器已經寫出床這個詞彙後，寫出y2（前）的機率
5. y3 是已經寫出床前這兩個詞彙後，寫出y3（明）的機率
6. until EOS



### 如何把 f 找出來？



### PixelRNN

* image generation : PixelRNN
* 不看黃色 pixel 而是看藍色的
* 是看上下的 pixel 而不是只看前一個



### Conditional Generation

* 可以寫詩的 RNN 沒辦法做出看一幅畫、寫一首詩
* 看到 young girl dancing, 生成影片描述
* chat bot：看到某句話、回說某句話



### Seq2Seq: 使用 encoder-decoder

* encoder: 把文本翻成「machine 的語言」
* decoder: 把 machine 的語言翻成人的語言



### Dynamic Conditional Generation

* 把 h1~h4 想成 database 裡面的資料
* 類似搜尋，搜 database 有沒有出現過的資料
* c 是由多個 hi 組成的，怎麼選出？
* decoder 自己選要哪些 hi 來組成 c



### Machine Translation - Attention Based

* 把中文機器學習翻成英文 machine learning
* match: 不要讀整句，只專注在某小段
* Google Translation



### Speech Recognition

* 可以把聲音轉成拼音，甚至可以拼出沒看過的單字



### Schedule Sampling

* （from reference）給機器看正確答案： train, test 不一致，一步錯、步步錯（因為沒看過）
* （from model）給機器看假的：會錯
* 結合兩者，擲骰子
* hw2-1 用到



### Beam Search

* greedy, 只選機率高的邊
* 不一定會 sampling，e.g. chatbot
* e.g. 用 argmax() 一層一層乘在一起，真的是機率最小ㄇ
* testing 才用到 beam search



### objective function 不能微分的話，就把它當 RL 的 reward 硬解



