---
title: 【快速入门机器翻译（二）】Tansformer 和 注意力机制
date: 2021-12-20 16:34:00
layout: python machine-translation machine-learning
---

Transformer模型在Google发布的论文“Attention is All You Need”中提出，现在是谷歌云TPU推荐的参考模型。（论文的地址:  [Attention is All You Need](https://arxiv.org/abs/1706.03762)）

# 从RNN到Tansformer
Transformer是一个 Seq2Seq 模型加上了注意力机制（Self-attention）。在Transformer中，self-attention可以直接被认为是一个layer，这个layer优化了RNN这种最传统的sequence处理方式，使之能够通过并行优化速度，并达到相同的sequence处理效果。下面我们从最开始的RNN开始，逐步介绍一下机器翻译的算法是如何一步一步走到transformer的。

机器翻译需要完成的任何是，将一个语言的句子映射成另一个语言的句子。那么它的输入必然是有序的，因为句子就是有序的。处理有序的数据，我们首先想到的是RNN。下图为一个典型的简易RNN模型，它的输入是一串sequence (a1,a2,a3,a4)，输出是另一串sequence (b1,b2,b3,b4)。

<img src="../../../../images/transformer1.png" style="height: 20em; margin: 20px;"/>

无论single direction的RNN还是bi-direction的RNN，每一个输出已经包含了所有输入的信息。目前听起来，RNN是非常符合机器翻译需求的模型，它能完成将一组语言序列映射到另一组语言序列的任务。但RNN有一个很致命的缺点，它很难被平行化计算。RNN的计算过程是顺序依赖的，这使得算法科学家很难通过并行的方式提高计算速度。而当你尝试将一门人类语言翻译成另一门人类语言时，计算量毫无疑问是巨大的，无法并行成为了一个很严重的问题。

为了将计算平行化，有人尝试使用CNN来解决机器翻译的问题。CNN可以通过叠加多层来使得每一个输出的b都包含所有a的计算，并且还能平行化计算。但CNN的问题在于，必须叠加很多层才能使得输出综合考虑所有输入，如果仅仅使用一层是无法达到效果的。

<img src="../../../../images/transformer2.png" style="height: 20em; margin: 20px;"/>

<br/>
接下来，Google提出了Self-attention层。这个layer不是一个RNN，但它跟双向RNN的输入输出一样，也是输入一个sequence输出另一个sequence，每个输出都包含了所有的输入信息，并且同时它能通过并行计算来优化速度。基本上，之前用RNN来做的东西，目前都有人尝试使用self-attsntion去取代它。

<img src="../../../../images/transformer3.png" style="height: 20em; margin: 20px;"/>


# Self-attention 计算过程
self-attention层是整个transformer中最重要的一部分。下面我将先介绍self-attention层是如何计算的，再介绍transformer的结构设计。

## Self-attention 内部的魔法

<img src="../../../../images/transformer4.png" style="height: 20em; margin: 20px;"/>

x1, x2, x3, x4是我们的输入。第一步是NLP的常规处理方式，我们做一个Embeding得到a1, a2, a3, a4。接下来我们分别将a乘上三个matrix得到三个新的vector，这三个vector分别叫q(query: to match others), k(key: to be matched), v(value: info to be exacted). 其中k代表key，表示一个match的符号；q表示query，用于match key；v表示需要被提取的信息/特征。

<img src="../../../../images/transformer5.png" style="height: 20em; margin: 20px;"/>

接下来我们用到了attention。我们将q1与所有的k分别做self-attention，得到了四个a (alpha)。self attention的计算方法在文章中使用的是dot-product，但这不是必须的，也可以更换为其他的计算方式。

<img src="../../../../images/transformer6.png" style="height: 20em; margin: 20px;"/>

接下来将四个alpha做一个SoftMax，并得到alpha hat。

<img src="../../../../images/transformer7.png" style="height: 20em; margin: 20px;"/>

然后将每个alpha hat与v做加权和，得到的就是b1。b1就是我们self attention 层的第一个输出。来看一下完整的b1计算过程。

<img src="../../../../images/transformer8.png" style="height: 20em; margin: 20px;"/>

## Self-attention 怎么并行计算
先复习一下self-attention层的输入输出。
<img src="../../../../images/transformer9.png" style="height: 20em; margin: 20px;"/>

然后我们将self-attention层的所有计算，列成矩阵运算，可以总结为下图。
<img src="../../../../images/transformer10.png" style="height: 20em; margin: 20px;"/>

不难看出，所有的运算都是矩阵运算，我们的计算机很轻易地就能优化矩阵运算。

## 如何引入顺序

回顾我们的self-attention层，可以发现一个很重要的问题，所有的输入输出的顺序没有被考虑进来。然而顺序（语序）对语言来说毫无疑问是重要的，那么我们怎么引入顺序呢？

论文中使用的Positional Encoding的方式，为每一个position设置了一个positional vector e, 这个vector e 是指定的而不是学习的。
<img src="../../../../images/transformer11.png" style="height: 20em; margin: 20px;"/>

# transformer 结构

了解了self-attention layer后，我们可以愉快地看懂transformation的结构啦！

<img src="../../../../images/transformer12.png" style="height: 30em; margin: 20px;"/>

如图所示，transformer是一个sequence to sequence结构。图的左半部分就是encoder，后半部分是decoder。先看左边的encoder，其中包含三种层：Multi-head attention、Add&Nor和Feed Forward。其中attention就是我们上面提到的，但Add & Nor和Feed Forward是什么呢？

## Encoder

<img src="../../../../images/transformer13.png" style="margin: 20px;"/>
上面这段学术地解释了Add & Nor是什么。如果看不懂，就认为Add就是指把上一层的输入和输出相加，得到一个新的vector，并把这个vector做一个normalized。

Feed Forward层我们也非常熟悉。它是一个两层的全连接层，第一层的激活函数为 Relu，第二层不使用激活函数，对应的公式如下。
<img src="../../../../images/transformer14.jpeg" style="with: 20em; margin: 20px;"/>

## Decoder

下面来看第一个attention层，这里有一个特殊的masked操作。因为在翻译的过程中是顺序翻译的，即翻译完第 i 个单词，才可以翻译第 i+1 个单词。通过 Masked 操作可以防止第 i 个单词知道 i+1 个单词之后的信息。

第二个 Multi-Head Attention计算没有变化，需要注意Self-Attention 的 K, V矩阵不是使用 上一个 Decoder block 的输出计算的，而是使用 Encoder 的输出编码信息矩阵 C 计算的。这样做的好处是在 Decoder 的时候，每一位单词都可以利用到 Encoder 所有单词的信息 (这些信息无需 Mask)。

## Transformer 总结
* Transformer 与 RNN 不同，可以比较好地并行训练。
* Self-Attention 本身是不能利用单词的顺序信息的，因此需要在输入中添加位置 Embedding，否则就是一个词袋模型了。
* Transformer 的重点是 Self-Attention 结构，其中用到的 Q, K, V矩阵通过输出进行线性变换得到。
* Transformer 中 Multi-Head Attention 中有多个 Self-Attention，可以捕获单词之间多种维度上的相关系数 attention score。

# Reference

[Attention is All You Need](https://arxiv.org/abs/1706.03762)

[Youtube: 李宏毅老师讲解transformer](https://www.youtube.com/watch?v=ugWDIIOHtPA&list=PLJV_el3uVTsOK_ZK5L0Iv_EQoL1JefRL4&index=62&ab_channel=Hung-yiLee)

[知乎专栏：初识CV的Transformer模型详解（图解最完整版）](https://zhuanlan.zhihu.com/p/338817680)