# Homework 3


## Lecture11

矩阵乘法如何加速：
1. 分块计算（一次计算出一块的结果）
2. 每次一行（列）的数据会被重复计算N次，可以放入L1 cache，减少访问时间。

## Q.1
`Reshape`只需修改`shape`的形状，然后根据形状重新计算`stride`，然后重新排布。
`Permute`需要根据排序后新的`shape`和`stride`，重新排布。
`broad_cast`将维度为1的部分`stride`设置为0。
`__getitem__`比较复杂，需要根据每一个维度的起点，首先计算出偏移量，然后根据每一个维度的起点，终点和步长计算出`shape`，并且将每一个维度的步长乘以`step`。


## Q.2
`Compact`将a的内容，通过`shape`,`offset`,`stride`取出对应的位置，然后按+1的顺序放入`out`中。

`EwiseSetitem`将`a`的内容按顺序取出，放入`out`的对应位置，`out`非`compact`，需要根据`stride` `offset`计算对应放入的位置。

`ScalarSetitem`同理，赋值变为常数。

## Q.3
已经假设`a`和`b`都是`compact`的，直接遍历一遍即可。

## Q.4
