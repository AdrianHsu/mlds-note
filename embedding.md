# Word Embedding



## 前置

假設整個 dataset 有以下句子：

1. I love you
2. love you
3. I love you all
4. I hate you



### 什麼是 padding?

這個 dataset 一共有 4 句，每一句的字數分別為 `(3, 2, 4, 3)`，那麽 padding 就會把每一句的長度補齊（`len = 4`），變成 `(4, 4, 4, 4)`。也就是變成

1. I love you <PAD>
2. love you <PAD> <PAD>
3. I love you all
4. I hate you <PAD>



### 什麼是 corpus?

語料庫，可以想成是一個 dict，大小為 `dict_size`。以此 dataset 為例，共有 `I, love, you, all, hate` 這五個字，所以 `dict_size` 就是 5。



### 這個矩陣長什麼樣子？

我們先看 I hate you <PAD> 這句話，把每個字做成 one-hot 形式會變成：

````
[1, 0, 0, 0, 0]
[0, 0, 0, 0, 1]
[0, 0, 1, 0, 0]
[0, 0, 0, 0, 0]
````

其中，第一個 row 代表 I，第二個 row 代表 hate，第三個 row 代表 you，第四個代表 <PAD>。所以我們可以把 dataset 中的每一句都轉成 `4 x 5` 也就是 `(len, dict_size)` 的二維矩陣。



### 整個 dataset 是什麼樣子？

就是把每個二維矩陣疊起來變成三維，也就是 `(dataset_size, len, dict_size)` 這個例子是 `(4, 4, 5)` 這樣就完成了。



## Embedding 是什麼？

我們接下來要考慮的問題是：如果 corpus 很多元怎麼辦？例如 `dict_size = 20000` 這很常見，因為每句話的組成太多樣了，那這個矩陣就會很大。這時候就會用到  **word embedding** ，也就是把 20000 維的 corpus 降維成 50 維之類的大小。我們可以用 **glove** 之類的 API 做到這件事：把 `dict_size` 降維成 50。可以想成是 embedding 就是一個大矩陣，然後跟原本矩陣相乘，被投影成小矩陣。`embed_matrix.shape` 就是 `(dict_size, 50)` 。





# 什麼是 tf.embedding_lookup() ?



```
params1 = tf.constant([1,2])
params2 = tf.constant([10,20])
ids = tf.constant([2,0,2,1,2,3])
result = tf.nn.embedding_lookup([params1, params2], ids)
```

index 0 corresponds to the first element of the first tensor: 1

index 1 corresponds to the first element of the second tensor: 10

index 2 corresponds to the second element of the first tensor: 2

index 3 corresponds to the second element of the second tensor: 20

Thus, the result would be:

```
[ 2  1  2 10  2 20]
```



所以 seq2seq 教學裡面的，

```python
# Embedding
embedding_encoder = variable_scope.get_variable(
    "embedding_encoder", [src_vocab_size, embedding_size], ...)
    
# Look up embedding:
#   encoder_inputs: [max_time, batch_size]
#   encoder_emb_inp: [max_time, batch_size, embedding_size]
encoder_emb_inp = embedding_ops.embedding_lookup(
    embedding_encoder, encoder_inputs)
```

這個 lookup 就是在 embedding__encoder 這個 list of tensors 裡面，找到並回傳 encoder_inputs 這些 ids 對應的降維後的 embedding vector。

[https://stackoverflow.com/questions/34870614/what-does-tf-nn-embedding-lookup-function-do](https://stackoverflow.com/questions/34870614/what-does-tf-nn-embedding-lookup-function-do)