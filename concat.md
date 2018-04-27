# tf.concat() 怎麼用

[https://blog.csdn.net/mao_xiao_feng/article/details/53366163](https://blog.csdn.net/mao_xiao_feng/article/details/53366163)

```python
t1 = [[1, 2, 3], [4, 5, 6]]  
t2 = [[7, 8, 9], [10, 11, 12]]  
tf.concat(0, [t1, t2]) == > [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]  

t1 = [[1, 2, 3], [4, 5, 6]]  
t2 = [[7, 8, 9], [10, 11, 12]]  
tf.concat(1, [t1, t2]) ==> [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]


# tensor t3 with shape [2, 3]  
# tensor t4 with shape [2, 3]  
tf.shape(tf.concat(0, [t3, t4])) ==> [4, 3]   # 就是上面第一個例子
tf.shape(tf.concat(1, [t3, t4])) ==> [2, 6]   # 第二個例子


t1=tf.constant([1,2,3])  
t2=tf.constant([4,5,6])  
# 因為它們對應的shape只有一個維度，當然不能在第二維上連了，雖然實際中兩個向量可以在行上連，但是放在程序裡是會報錯的
#concated = tf.concat(1, [t1,t2])這樣會報錯  
t1=tf.expand_dims(tf.constant([1,2,3]),1)  
t2=tf.expand_dims(tf.constant([4,5,6]),1)  
concated = tf.concat(1, [t1,t2])#這樣就是正確的  
```

