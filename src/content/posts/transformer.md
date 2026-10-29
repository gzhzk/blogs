---
title: "Transformer 核心架构逐层拆解"
published: 2026-03-02
updated: 2026-03-07
description: "从 Self-Attention 出发，逐层拆解 Transformer 的编码器、缩放点积注意力、多头注意力、掩码与位置编码，配合公式、架构图和 PyTorch 实现。"
tags: ["Transformer", "Attention", "PyTorch"]
category: "算法与组件"
lang: "zh_CN"
draft: false
---

**Transformer 架构** 是2017年由 Vaswani 等八位 Google 大佬在论文《Attention is All You Need》中提出的**基于自注意力机制（Self-Attention）的序列建模架构**，它完全抛弃了循环结构，转而完全依赖一种名为注意力（Attention）的机制来捕捉序列内的依赖关系，从而实现了真正意义上的**并行计算**。它彻底改变了自然语言处理领域（Natural Language Processing, NLP）及其他序列任务的方式，对当今大语言模型（Large Language Model, LLM）的发展做出了跨时代的贡献。借此博客，分享一下我对 Transformer 架构的理解。希望对你有帮助！

## Why Self-Attention?

当今大语言模型的一大核心任务是处理长文本，专业一点的说法是序列转档。在 Transformer 架构提出之前，传统的序列转档模型依托于**循环神经网络（Recurrent Neural Network, RNN）**，它的原理并不复杂。例如，当你阅读这篇文章时，你对每个词的理解都建立在对前面词的理解之上，RNN 这种带有循环的网络，能够让信息得以保留（**图1**）。从中可以直观看出 RNN 的核心计算思路是将模型在上一时间步产生的隐藏状态作为当前时间步输入的一部分，与当前的外部输入共同计算新的隐藏状态和输出。通过循环计算更新隐藏状态，再将这个状态传递给下一个时间步，如此逐步推进处理整个序列（由于具体原理篇幅较长，后续我会单独整理一篇关于 RNN 和 LSTM 模型的理解）。

然而，这种逐步传递的方式存在显著缺陷：由于信息必须按时间顺序逐层传递，模型 **难以捕捉长距离依赖**（距离越远的信息衰减越严重），且**无法并行计算**（必须等上一步完成后才能进行下一步），导致训练效率低下。为了更直观地理解这个问题，不妨想象你正在做英语的阅读理解，通常有两种做题策略：一种是从头到尾逐字逐词阅读，必须读完前面才能慢慢理解后面的内容，如果文章很长，你可能会忘记开头的关键信息，导致做题时难以快速定位答案；而另一种则是先读问题，**带着问题去扫描文章**，通过关键词快速找到最相关匹配度最高的段落或某一行，再提取其中的关键信息，这样无论文章多长，你都能直接跳到最高度相关的地方，既高效又能捕捉全局语义。

或许正是受到这一想法的启发，Transformer 才引入了自注意力机制，让模型能够同时关注序列中的所有位置，直接建立任意两个位置之间的依赖关系，从而实现对长文本的高效并行建模。

![RNN展开结构示意图](/blogs/images/1/RNN.png)

**图1: RNN展开结构示意图**

>>> 左侧为一个循环单元"U"，接收输入x<sub>t</sub>和前一时刻隐藏状态h<sub>t-1</sub>，输出当前隐藏状态h<sub>t</sub>。右侧展示该单元沿时间轴的展开，揭示了RNN处理序列数据的方式。

## Transformer 整体架构：编码器（Encoder）+ 解码器（Decoder）

在上面我们已经了解到，自注意力机制能让序列模型突破 RNN 的“串行局限”，起到**同时关注整个序列，动态捕捉长距离依赖**的关键作用，这将是 Transformer 架构的核心组成部分。但是，为了完成更复杂的任务，还需要考虑到方方面面的其他因素，其中有一点：计算机无法直接理解我们人类的语言，它最底层的结构决定了它只能理解数字、运算数字。那么，就需要通过某种方法将人类语言转换成机器可以理解的“机器语言”，而且还要保证“机器语言”带有我们想要表达的信息（即语义）。没错，在数学中的的确确就有这样一种既使用数字表示还能刻画特征的东西—— **向量（Vector）**（具体原理涉及到自然语言处理中文本表征的相关知识，详见我的上一篇博客《NLP 文本表征：Word Embedding + Tokenizer + BPE 算法全解》）。

在 **Transformer** 的整体架构设计中（**图2**），原作者设计了这两大模块：**编码器（Encoder）+ 解码器（Decoder）**。从功能上看，编码器是深度理解，将源序列转化为一个**上下文感知的向量序列**，并通过一系列复杂的向量运算在不断训练中逐渐提取出语义信息；而解码器是**条件生成**，一边参考这些语义向量，一边逐步生成出目标序列。这种结构天然适配机器翻译、文本摘要等序列到序列任务，也正是当前大语言模型的核心基础。接下来，我们一个模块一个模块拆解，感受作者天才般的智慧。

![Transformer 模型整体架构示意图](/blogs/images/1/Transformer.png)

**图2: Transformer 模型整体架构示意图**

>>> 左侧为编码器（Encoder），右侧为解码器（Decoder），均为 N 层堆叠。输入侧以 Embedding 结合位置编码完成向量化；编码器通过自注意力与 FFN 提取上下文语义；解码器借助掩码自注意力 + 交互注意力 + FFN 生成目标序列；最后经线性层和 Softmax 输出预测概率。

## 编码器结构：多头注意力、残差连接、层归一化、前馈网络

### 核心创新点1：Scaled Dot-Product Attention

想象你正在做一篇英语阅读理解，有一个让做题变得又快又准的策略——带着问题找答案，这往往能帮助我们快速锁定文本中的关键信息，可能对着一篇上千词的文段，快速扫几眼就解决了，这其实就体现了注意力的分配。然而对于计算机，却难以快速理解。因此，我们需要一个机制，让模型能够基于输入动态地决定关注哪里。就像数据库查询：给定一个问题（Q），在所有记录的关键字（K）中查找匹配，然后返回对应的值（V）。下面的 **图3** 呈现了注意力机制的精髓：**缩放点积注意力（Scaled Dot-Product Attention）**。

![缩放点积注意力（Scaled Dot-Product Attention）结构图](/blogs/images/1/QKV.png)

**图3: 缩放点积注意力（Scaled Dot-Product Attention）结构示意图**

>>> 这是 Transformer 模型中的核心注意力计算机制，其中左侧输入Q（Query）和K（Key），右侧输入V（Value）。箭头指示了从Q、K计算相似度分数，经缩放、可选的掩码、SoftMax归一化，最后与V加权求和的数据流向。

**注意力机制的核心思想是对齐（Alignment）**，即同一序列内(即自注意力或不同序列间(即交叉注意力)词语之间的关系。注意力机制本质上是在计算序列中每个位置与其他所有位置的相关性分数。这种对齐不是简单的单词匹配，而是语义和语法层面的深度关联。

注意力机制的工作过程是这样的：

（1）**计算注意力权重**：首先明确一点，Q、K、V 三个矩阵都是利用原矩阵 X 做线性变换得到的，如 **图4** 所示。得到这三个矩阵后，（先考虑单个 batch 的情况，形状是 seq_len × d_model，即每个矩阵的行表示序列中的每一个 token，列表示每个 token 对应的特征），为了使语义对齐，向量中的点积概念开始登场。其实很自然地，我们可以认为序列中的某一个 token(A) 的行向量q 去点乘其他所有 token(B, C, D, etc) 的列向量k 从而获得一个分数，这里分数的意思很明确，就是表示 token(A) 与其他所有 token 的相关性。比如按照 **图5** 中的例子，用第三行的 went 的 q 向量（包含了 went 的所有查询语义特征，如'我带有去某个地方的语义，需要关注地点名词'、'我往往表示人或物所做的动作，需要关注人或物'）去与 The，group，home 的 k 向量的转置做向量点积运算。在计算点积时，went 的'需要地点'特征与 home 的'是地点'特征相乘得到高分，went 的'需要主语'特征与 group 的'是主语'特征相乘得到高分，这些高分的总和使得 went 与 home 和 group 的相关性都强。

但是，这里有一个逻辑漏洞，点积将高维向量的所有维度信息压缩成一个标量，我们好像无法从这个单一得分中获知是哪些具体的语义特征（如“地点”或“主语”）主导了相关性判断。在前向传播时，点积确实是一个“黑箱”操作，输出一个代表整体相关性的分数。模型此刻并不关心里面0.73是0.72（地点维度贡献）+0.01，还是别的什么组合。怎么办呢？真正的魔法其实发生在反向传播。它按照梯度方向更新可学习的参数矩阵 W_q 和 W_k。在无数次的迭代中，一种高效的优化路径会逐渐浮现：要想显著提高点积总分，最有效的方法是让 q_went 在地点相关维度上获得高权重，同时让 k_home 在地点维度上也获得高权重。因为当这两个向量在地点维度上同时取较大值时，它们的乘积会对最终总和产生决定性影响。于是，通过梯度下降的“压力”，W_q 和 W_k 被调整，尽管点积计算仍输出一个混合值0.73，但其内部构成已被学习过程塑造成由“地点”维度主导。这也就解释了为什么 图4 中我提到变换矩阵都是学习得到的。

![QKV由原矩阵X经过线性变换结构图](/blogs/images/1/attention-encoder-self-attention-qkv-transformation.png)

**图4: Q/K/V由原矩阵X经过线性变换示意图**

图源：
[chrisvdweth/selene/attention-encoder-self-attention-qkv-transformation.png](https://github.com/chrisvdweth/selene/blob/master/notebooks/images/illustrations/attention/attention-encoder-self-attention-qkv-transformation.png)

这是得到 Q、K、V 矩阵的关键步骤，对原矩阵 X 做线性变换，从而使原矩阵具备了不同的功能（查询、键、值），方便后续计算注意力的处理。注意，这里的变换矩阵 W_q、W_k、W_v 都是后续学习得到的。

![Q与K的转置做点积运算图](/blogs/images/1/attention-encoder-self-attention-scores.png)

**图5: Q与K的转置做点积运算示意图**

图源：
[chrisvdweth/selene/attention-encoder-self-attention-scores.png](https://github.com/chrisvdweth/selene/blob/master/notebooks/images/illustrations/attention/attention-encoder-self-attention-scores.png)

$$
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V
$$

```python
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    缩放点积注意力
    参数含义：
        Q：查询张量，shape (batch_size, num_heads, seq_len_q, d_k)
        K：键张量，shape (batch_size, num_heads, seq_len_k, d_k)
        V：值张量，shape (batch_size, num_heads, seq_len_k, d_v)  一般情况下，d_v = d_k
        mask：掩码张量，shape 可与 scores 广播，布尔类型，True 的位置将被掩盖（设为 -inf）
    返回：
        output：注意力输出，shape (batch_size, num_heads, seq_len_q, d_v)
        attention_weight：注意力权重，shape (batch_size, num_heads, seq_len_q, seq_len_k)
    """
    # 获取最后一个维度的大小，即每个注意力头的特征维度，缩放时使用
    d_k = Q.size(-1)
    # 计算 Q 与 K 的点积，得到的scores形状：(batch_size, num_heads, seq_len_q, seq_len_k)
    # 并缩放，缩放保持方差稳定，使训练更稳定，防止梯度消失
    scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))
    print(scores.shape)
    # 应用掩码
    if mask is not None:
        # mask是布尔张量，True的位置需要被掩盖，掩码位置的值设为负无穷
        scores = scores.masked_fill(mask, float('-inf'))    
    # 经过 softmax 得到注意力权重，dim=-1表示在最后一个维度（seq_len_k）上进行softmax
    attention_weight = F.softmax(scores, dim=-1)
    # 加权求和得到输出
    output = torch.matmul(attention_weight, V)

    return attention_weight, output

if __name__ == "__main__":
    batch_size = 2
    num_heads = 1
    seq_len = 4
    d_k = 8
    d_v = d_k

    # 随机生成 Q, K, V (batch_size, num_heads, seq_len, d_k)
    torch.manual_seed(42)
    Q = torch.randn(batch_size, num_heads, seq_len, d_k)
    K = torch.randn(batch_size, num_heads, seq_len, d_k)
    V = torch.randn(batch_size, num_heads, seq_len, d_v)

    attention_weight, output = scaled_dot_product_attention(Q, K, V)
    print(attention_weight)
    print(output)

    """
        填充掩码 Padding Mask
        构造 padding mask：有效位置为 False，填充位置为 True（需要掩盖）
        在 batch 处理中，不同序列往往长度不同，需要用 padding 来补齐到与最长句子相同的长度
        填充掩码的作用是：防止模型学习无意义的填充符；提高并行时的计算效率
        假设序列长度的真实值： batch 0 的前3位有效，batch 1的前2位有效
    """
    valid_lens = torch.tensor([3, 2])
    padding_mask = torch.arange(seq_len).expand(batch_size, seq_len) >= valid_lens.unsqueeze(1)
    padding_mask = padding_mask.unsqueeze(1).unsqueeze(2)
    attn_weight_pad, output_pad = scaled_dot_product_attention(Q, K, V, mask=padding_mask)
    print(attn_weight_pad)
    print(output_pad)
    
    """ 
        因果掩码 Casual Mask
        构造 casual mask：有效位置为 False，填充位置为 True（需要掩盖）；在序列处理中，后续位置需要掩盖，防止模型看到未来的信息
        因果掩码的作用是：保证生成第t个词时，只能基于前t-1个词；训练时模拟推理时的条件
        生成上三角矩阵，对角线以上为 True（需要掩盖）
    """
    casual_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    casual_mask = casual_mask.unsqueeze(0).unsqueeze(0)
    attn_weight_causal, output_causal = scaled_dot_product_attention(Q, K, V, mask=casual_mask)
    print(attn_weight_causal)
    print(output_causal)
```

### 核心创新点2：Multi-Head Attention

![多头注意力（Multi-Head Attention）结构图](/blogs/images/1/VKQ.png)

**图5: 多头注意力（Multi-Head Attention）结构示意图**

在多头自注意力机制中，输入的V（Value）、K（Key）和Q（Query）向量分别经过多个独立的"Linear"线性变换模块，然后并行进入多个"Scaled Dot-Product Attention"模块进行计算。各注意力头的输出通过"Concat"模块拼接，最后再经过一个"Linear"模块进行线性变换融合。箭头清晰地显示了从输入到输出的整个数据流动过程。

$$
\begin{aligned}
\text{MultiHead}(Q, K, V) &= \operatorname{Concat}\left( \text{head}_{1}, \ldots, \text{head}_{h} \right) W^{O} \\
&\text{ where } \text{head}_{i} = \operatorname{Attention}\left( Q W_{i}^{Q}, K W_{i}^{K}, V W_{i}^{V} \right)
\end{aligned}
$$

## 解码器结构：掩码注意力、交叉注意力、解码流程

## 位置编码：给序列注入位置信息

$$
\begin{aligned}
PE_{(pos,2i)} &= \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right) \\
PE_{(pos,2i+1)} &= \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
\end{aligned}
$$

## 编码器与解码器适用场景：BERT系、GPT系架构区分
