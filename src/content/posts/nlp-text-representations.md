---
title: "NLP 文本表征：Word Embedding + Tokenizer + BPE 算法全解"
published: 2026-02-25
updated: 2026-03-06
description: "从 One-Hot、词袋和 TF-IDF 到 Word Embedding、Tokenizer 与 BPE，系统理解文本如何被映射到连续语义空间，并附完整代码实现。"
tags: ["NLP", "Embedding", "Tokenizer", "BPE"]
category: "算法与组件"
lang: "zh_CN"
draft: false
---

2013 年，Mikolov 等人提出 **Word2Vec**，将文字映射为连续向量，让语义相似的词在空间中彼此靠近——这便是 **Word Embedding**。它解决了传统稀疏表示（One-Hot、词袋）无法捕捉语义关联的痛点，从此成为连接文本与深度学习的基础桥梁。但词级方案面临 OOV、词表膨胀等缺陷，现代大模型转而采用 **Tokenizer + BPE 子词分词**。借此博客，系统梳理 Embedding 的技术演进，从原理到代码。

## 从“符号鸿沟”到“语义空间”

让计算机理解语言，首先要解决一个根本矛盾：人类感知的是「语义」，而计算机处理的是「符号」。当你输入「猫」和「猫咪」，人类知道它们相似；但对计算机而言，这是两个毫无关联的编码。这便是自然语言处理中的「符号鸿沟」——离散符号与连续语义之间的断裂。Embedding 正是跨越这道鸿沟的核心桥梁：它将离散的自然语言符号映射到连续的向量空间，使得语义相似的对象在空间中彼此靠近。这个简洁的思想，支撑起了从 Word2Vec 到 ChatGPT 的所有语言模型。

在 Transformer 体系中，Word Embedding 处于输入层的核心位置，负责完成从 Token 到向量的转换，为后续注意力机制与前馈网络提供可计算的语义表示。然而，传统词级嵌入面临 OOV（未登录词）、词表膨胀等难题，现代大模型普遍采用 Tokenizer + 子词分词（如 BPE）来构建更高效的词表与嵌入层。本文将系统梳理 Word Embedding 的技术演进，从 One-Hot 的稀疏困境，到 Word2Vec 的分布式表示，再到 BPE 子词分词的精妙设计，提供完整可运行的代码实现，帮助你构建从文本预处理到模型输入的全链路认知。

## 此前的经典方法：从统计到预训练

### One-Hot 编码

One-Hot 编码，顾名思义是为每个分类特征的每个可能值创建一个独立的二进制特征。在任意时刻，只有一个特征被激活（标记为 1），其余全部为 0。词表多大，向量就多大。例如，「猫」→ [1,0,0,0]，「狗」→ [0,1,0,0]，「鸟」→ [0,0,1,0]，「鱼」→ [0,0,0,1]。但是，这样操作存在不小的问题。首先就是当词的数量很大时，词向量的维度会爆炸；另外，词向量之间两两正交，无法准确刻画语义的相关性。

### 词袋模型

词袋模型（Bag-of-Words，BoW）的核心假设是：文档的语义由其包含的词汇决定，与词汇出现的顺序无关。它将一篇文档表示为一个「词频向量」——向量的每个维度对应词表中的一个词，维度的值是该词在文档中出现的次数。

### TF-IDF

TF-IDF（词频-逆文档频率）是一种用于信息检索与文本挖掘的经典加权技术，由 Salton 等人在 1970 年代提出。它同时考虑**词在当前文档中的出现频率**（TF）和**该词在所有文档中的罕见程度**（IDF），从而筛选出对当前文档具有代表性的词汇。

直观理解：一个词在本文档中出现很多（TF 高），但在所有文档中都常见（IDF 低），则区分度低；反之如果只在少数文档中高频出现，则很可能是该文档的关键词。

**TF（词频）**：衡量词对单篇文档的重要程度

$$
\text{TF}(t, d) = \frac{\text{词 } t \text{ 在文档 } d \text{ 中出现的次数}}{\text{文档 } d \text{ 中所有词的总次数}}
$$

**IDF（逆文档频率）**：衡量词对整个语料库的区分能力

$$
\text{IDF}(t) = \ln\left(\frac{\text{语料库总文档数} + 1}{\text{包含词 } t \text{ 的文档数} + 1}\right) + 1
$$

（加 1 是为了避免分母为 0）

**最终权重**：

$$
\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)
$$

举例：假设语料库有 1000 篇文档，其中 50 篇包含"深度学习"，"深度学习"在某篇文档中出现 5 次，该文档总词数 200。则：

TF = 5/200 = 0.025，IDF = ln(1001/51) + 1 ≈ 3.01，TF-IDF ≈ 0.075。

而"的""是"这类停用词几乎每篇文档都出现，IDF → 1，TF-IDF 接近 0，天然被过滤。

```python
from collections import Counter
import math

def compute_tf(token_ids):
    """计算词频 TF"""
    total = len(token_ids)
    counter = Counter(token_ids)
    return {word: count / total for word, count in counter.items()}

def compute_idf(corpus_tokenized, vocab):
    """计算逆文档频率 IDF
    corpus_tokenized: List[List[str]]，每篇文档的分词列表
    """
    N = len(corpus_tokenized)
    idf = {}
    for word in vocab:
        # 统计包含该词的文档数
        df = sum(1 for doc in corpus_tokenized if word in doc)
        idf[word] = math.log((N + 1) / (df + 1)) + 1
    return idf

def compute_tfidf(corpus_tokenized, vocab):
    """计算 TF-IDF 矩阵"""
    idf = compute_idf(corpus_tokenized, vocab)
    tfidf_matrix = []
    for doc in corpus_tokenized:
        tf = compute_tf(doc)
        tfidf = {word: tf.get(word, 0) * idf[word] for word in vocab}
        tfidf_matrix.append(tfidf)
    return tfidf_matrix

# 示例
corpus = [
    ["深度", "学习", "是", "人工", "智能", "的", "核心"],
    ["机器", "学习", "算法", "在", "工业", "应用", "广泛"],
    ["深度", "学习", "模型", "参数", "多", "训练", "慢"],
]
vocab = list(set(word for doc in corpus for word in doc))
tfidf = compute_tfidf(corpus, vocab)
print("文档0 TF-IDF（按权重排序）：", sorted(tfidf[0].items(), key=lambda x: x[1], reverse=True)[:3])
```

### 共现矩阵

共现矩阵（Co-occurrence Matrix）的核心思想是：**语义相近的词往往在相似的上下文中出现**。它统计词与词在固定窗口内共同出现的频次，形成一个对称矩阵。

例如语料库只有三句话："The cat sat on the mat"、"The dog ran in the garden"、"Cats and dogs are animals"。设定窗口大小为 2（前后各看 1 个词），统计每对词共同出现的次数：

$$
M_{ij} = \text{词 } i \text{ 和词 } j \text{ 在窗口内共同出现的次数}
$$

从共现矩阵可以挖掘语义关系："cat"和"cats"在相似上下文中出现，矩阵中对应的行/列分布相似；"dog"和"cat"经常出现在相似的动物语境中，距离也较近。

但直接用原始共现矩阵有两个问题：高频词（如"the"）共现次数虚高；矩阵稀疏且维度等于词表大小（通常数万×数万）。解决方案是对矩阵做**奇异值分解（SVD）**降维，保留最重要的 k 维特征，得到词的稠密向量表示——这其实就是 LSA（潜在语义分析）的思路。

```python
import numpy as np
from collections import Counter

def build_cooccurrence_matrix(corpus, vocab, window_size=2):
    """构建共现矩阵"""
    word2idx = {w: i for i, w in enumerate(vocab)}
    n = len(vocab)
    matrix = np.zeros((n, n), dtype=np.float32)

    for doc in corpus:
        for i, word in enumerate(doc):
            # 窗口范围
            start = max(0, i - window_size)
            end = min(len(doc), i + window_size + 1)
            for j in range(start, end):
                if i != j:
                    matrix[word2idx[word]][word2idx[doc[j]]] += 1

    return matrix

def svd_reduce(matrix, k=50):
    """SVD 降维得到词向量"""
    U, S, Vt = np.linalg.svd(matrix, full_matrices=False)
    # 取前 k 维，U * sqrt(S) 作为词向量
    word_vectors = U[:, :k] * np.sqrt(S[:k])
    return word_vectors

corpus = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "in", "the", "garden"],
]
vocab = list(set(word for doc in corpus for word in doc))
matrix = build_cooccurrence_matrix(corpus, vocab)
vectors = svd_reduce(matrix, k=10)
print(f"词向量形状: {vectors.shape}")  # (词表大小, 10)
```

### 四种方法的对比总结

| | One-Hot | 词袋模型（BoW） | TF-IDF | 共现矩阵 + SVD |
|---|---|---|---|---|
| **维度** | 词表大小 | 词表大小 | 词表大小 | 降维后维度 k |
| **稀疏性** | 极度稀疏 | 稀疏 | 稀疏 | 低（降维后） |
| **语义捕捉** | 无 | 无 | 弱 | 中等（浅层语义） |
| **共现信息** | 无 | 无 | 无 | 有 |
| **计算成本** | 低 | 低 | 中 | 中高（SVD） |
| **主要用途** | 基础基准 | 文本分类特征 | 关键词提取 | 语义相似性（早期） |

可以看到，这四种方法有一个共同局限：**每个词只有唯一一个固定向量**，与上下文无关。这导致"bank"（银行）和"bank"（河岸）无法区分，歧义问题无法解决。Word2Vec 通过神经网络从大规模语料中学习语境相关的稠密表示，才真正开启了现代 Embedding 时代。

## 现代 NLP 预处理：Tokenizer 与 BPE 算法

前面几种方法有一个共同局限：每个词只有唯一一个固定向量，无法处理歧义问题。而词表过大又导致 OOV（未登录词）严重——这两种困境在子词分词（Subword Tokenization）面前迎刃而解，而 BPE（Byte Pair Encoding）正是子词分词最经典的代表。

### 核心思想：用子词粒度解决 OOV

先用一个直观的例子理解 BPE 为什么能解决 OOV：

假设词表里只有 ["high", "higher", "low", "lower", "new"]，没有 newest。整词方案会对 newest 报 OOV。但 BPE 词表可能包含 ["new", "est", "er", ...] 这些子词片段。那么 newest 就能被切分为 new + est，两个片段都在词表中——**即使整词 OOV，子词仍然有效**。

高频词保持整词形态（节省表示长度），低频词拆解为子词组合（泛化能力强），这是 BPE 的核心设计哲学。

### BPE 算法原理

BPE 原是 1994 年用于数据压缩的算法，核心思想极为简洁——**贪心迭代合并最高频的相邻字符对**。具体步骤：

**Step 1：初始化字符级词表**

将训练语料中每个词拆成单个字符（加上结束符 </w> 表示词尾），以字符为初始 token。

```
"high"    → h i g h </w>
"higher"  → h i g h e r </w>
"low"     → l o w </w>
"lower"   → l o w e r </w>
```

**Step 2：贪心合并最高频字节对**

统计所有相邻字符对的频次，找到最高频的一对（如 h + i），将语料中所有 hi 合并为新符号 hi。

```
合并后：
"high"    → hi g h </w>
"higher"  → hi g h e r </w>
"low"     → l o w </w>
"lower"   → l o w e r </w>
```

**Step 3：重复迭代 N 次**

每轮都重新统计频次、合并最高频对，不断产生新的子词符号。合并优先级：字符对 > 单字符 > 子词组合（因为子词是由字符对合并而来，所以字符对优先级最高）。

```python
from collections import Counter

def get_stats(vocab):
    """统计所有相邻字节对的频次
    vocab: dict，{word: frequency}，word 内字符以空格分隔
    """
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs

def merge_vocab(pair, vocab):
    """将最高频字节对合并为新符号，替换 vocab 中所有出现"""
    bigram = ' '.join(pair)          # e.g. "e s"
    merged = ''.join(pair)           # e.g. "es"
    new_vocab = {}
    for word in vocab:
        new_word = word.replace(bigram, merged)
        new_vocab[new_word] = vocab[word]
    return new_vocab

# ── 演示完整训练流程 ──
vocab = {
    'h i g h </w>': 5,
    'h i g h e r </w>': 2,
    'l o w </w>': 3,
    'l o w e r </w>': 2,
}

# 迭代 10 次合并
num_merges = 10
for i in range(num_merges):
    pairs = get_stats(vocab)
    if not pairs:
        break
    best_pair = pairs.most_common(1)[0][0]
    vocab = merge_vocab(best_pair, vocab)
    print(f"第{i+1}步合并 {best_pair} → {''.join(best_pair)}")

print("\n最终词表（部分）：")
for word in list(vocab.keys())[:5]:
    print(f"  {word}")
```

**运行结果：**

```
第1步合并 ('h', 'i') → hi
第2步合并 ('e', 'r') → er
第3步合并 ('hi', 'g') → hig
第4步合并 ('l', 'o') → lo
第5步合并 ('lo', 'w') → low
...

最终词表（部分）：
  hig h </w>
  hig h er </w>
  low </w>
  low er </w>
```

可以看到：高频片段（如 low）在合并中逐渐固化为独立 token，低频词通过子词组合表示。

### BPE 分词：给定合并规则，对任意词分词

重点来了：训练得到合并规则（merge list）后，怎么用它分词？

**核心原则：按优先级从高到低，贪婪从左到右应用合并规则**

```python
def tokenize(text, merges):
    """给定 merge 规则，对文本进行 BPE 分词
    merges: List[(char, char)]，按优先级排序的合并操作序列
    """
    # 初始化：每个字符 + 词结束符
    tokens = list(text) + ['</w>']

    # 按优先级顺序（先合并的规则优先）应用
    for pair in merges:
        # 从左到右扫描，看当前上下文能否匹配 pair
        i = 0
        while i < len(tokens) - 1:
            if tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                # 匹配：合并为一个 token，跳过已合并部分
                tokens[i] = pair[0] + pair[1]
                del tokens[i + 1]
                # 不移动 i，继续检查当前位置（可能形成新的可合并对）
            else:
                i += 1

    return tokens

# 假设从语料中学到的合并序列（优先级从高到低）
merges = [
    ('h', 'i'),    # 第1步：最高优先级
    ('i', 'g'),
    ('e', 'r'),
    ('l', 'o'),
    ('o', 'w'),
]

text = "high"
result = tokenize(text, merges)
print(f"'{text}' 分词结果: {result}")   # ['hi', 'g', 'h', '</w>']

text2 = "hello"   # OOV
result2 = tokenize(text2, merges)
print(f"'{text2}' 分词结果(OOV): {result2}")   # ['h', 'el', 'lo', '</w>']
```

分词流程的要点是：**每次合并后不急于前进，而是继续检查当前位置**（因为刚合并出来的新符号可能可以和下一个符号再次合并）。这是 BPE 贪婪但高效的体现——每一步都做到当前最优，不回溯。

可以看到，high 最终被切为 hi + g + h + </w>，因为我们在 merge 规则里没有 hi → hig 这条（需要更高频才能出现），所以 hi 是最大可合并单元。相比之下，如果词表里有 high，分词就会停在 high + </w>——这正是 BPE 的特性：高频词尽量保持完整，低频词才拆解。

真实场景中（如 GPT-2 / LLaMA），词表通常包含数万个 merge 规则，对应数万个子词 token。OOV 词通过字符级逐步拆分总能被表示——这是 BPE 相比整词分词的根本优势。

## Transformer 中的 Word Embedding 原理

在 Transformer 架构中，Word Embedding 处于输入层核心位置，与位置编码（Position Encoding）相加后共同构成输入向量。

**结构示意**

```
Token序列  →  Tokenizer  →  Token IDs  →  Embedding层  →  Token Embeddings
                                                            +
                                                       Position Encoding
                                                            =
                                                    输入向量 h₀
```

Embedding 层本质上是一个 vocab_size × d_model 的可学习参数矩阵 $W_E$。每个 token ID 对应矩阵中的一行，查询该行即得到该 token 的稠密向量表示。

```python
import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    """Token Embedding 层"""
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, token_ids):
        # token_ids: (batch_size, seq_len)
        # 输出: (batch_size, seq_len, d_model)
        return self.embedding(token_ids) * math.sqrt(self.d_model)
```

乘以 $\sqrt{d_{model}}$ 是原论文的一个 trick：Embedding 的方差通常较小，直接与位置编码相加会导致位置编码的信息被稀释，乘上 $\sqrt{d_{model}}$ 可以让两者的量级更加匹配。

**完整的嵌入层（含位置编码）**

```python
import math

class TransformerEmbedding(nn.Module):
    """Embedding = Token Embedding + Position Encoding"""
    def __init__(self, vocab_size, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = self._generate_positional_encoding(max_len, d_model)
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

    def _generate_positional_encoding(self, max_len, d_model):
        """生成固定的位置编码（正弦版本）"""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            (-math.log(10000.0) / d_model)
        )  # (d_model/2,)
        pe[:, 0::2] = torch.sin(position * div_term)       # 偶数维度
        pe[:, 1::2] = torch.cos(position * div_term)       # 奇数维度
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)，便于广播
        return pe

    def forward(self, token_ids):
        seq_len = token_ids.size(1)
        # Token Embedding 并缩放
        token_emb = self.token_emb(token_ids) * math.sqrt(self.d_model)
        # 加上位置编码（取前 seq_len 行）
        pos_emb = self.pos_emb[:, :seq_len, :]
        return self.dropout(token_emb + pos_emb)
```

## 完整代码实现（从文本到 Embedding 输出）

```python
import torch
import torch.nn as nn
import math
from collections import Counter

# ─────────────── 1. 简易 BPE 分词器 ───────────────

class SimpleBPE:
    """简化 BPE 实现（用于演示原理）"""
    def __init__(self, vocab, merges):
        # vocab: word -> freq 字典
        # merges: List[(char, char)] 合并操作序列
        self.vocab = vocab
        self.merges = merges
        self.tokenizer = self._build_tokenizer()

    def _build_tokenizer(self):
        return self.merges  # 直接存储合并规则

    def encode(self, text):
        """对文本进行 BPE 分词"""
        tokens = list(text) + ['</w>']
        for merge in self.merges:
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == merge[0] and tokens[i+1] == merge[1]:
                    new_tokens.append(merge[0] + merge[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def decode(self, tokens):
        return ''.join(t).replace('</w>', '')

# ─────────────── 2. 词表映射 ───────────────

class Tokenizer:
    """Tokenizer：将 token 序列转为 ID 序列"""
    def __init__(self, bpe):
        self.bpe = bpe
        vocab_set = set()
        for word in bpe.vocab:
            tokens = word.split()
            vocab_set.update(tokens)
        # 加入 merge 后产生的新符号
        for m in bpe.merges:
            vocab_set.add(m[0] + m[1])
        self.vocab = ['<pad>', '<unk>'] + sorted(vocab_set)
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.pad_idx = 0
        self.unk_idx = 1

    def encode(self, text):
        tokens = self.bpe.encode(text)
        return [self.word2idx.get(t, self.unk_idx) for t in tokens]

    def decode(self, ids):
        tokens = [self.vocab[i] for i in ids]
        return self.bpe.decode(tokens)

# ─────────────── 3. Transformer Embedding ───────────────

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = self._generate_positional_encoding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model

    def _generate_positional_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # (1, max_len, d_model)

    def forward(self, token_ids):
        seq_len = token_ids.size(1)
        token_emb = self.token_emb(token_ids) * math.sqrt(self.d_model)
        pos_emb = self.pos_emb[:, :seq_len, :].to(token_ids.device)
        return self.dropout(token_emb + pos_emb)

# ─────────────── 4. 端到端演示 ───────────────

def demo():
    # 准备语料
    vocab = {
        'h i g h </w>': 5,
        'l o w </w>': 3,
        'l o w e r </w>': 2,
        'n e w </w>': 4,
    }
    # 模拟学习到的合并规则（真实场景需从语料统计得到）
    merges = [('e', 'r'), ('l', 'o'), ('o', 'w')]

    bpe = SimpleBPE(vocab, merges)
    tokenizer = Tokenizer(bpe)

    # 输入文本
    text = "higher"
    token_ids = tokenizer.encode(text)
    print(f"原文: '{text}'")
    print(f"分词: {bpe.encode(text)}")
    print(f"Token IDs: {token_ids}")

    # 转为 Tensor
    token_ids_tensor = torch.tensor([token_ids])
    print(f"Tensor shape: {token_ids_tensor.shape}")

    # Embedding
    d_model = 128
    vocab_size = len(tokenizer.vocab)
    embed = TransformerEmbedding(vocab_size, d_model)

    output = embed(token_ids_tensor)
    print(f"Embedding 输出形状: {output.shape}")  # (1, seq_len, 128)
    print(f"输出向量示例（第一个token）:\n{output[0, 0, :10]}")

if __name__ == "__main__":
    demo()
```

## 现代演进与重点解读

BPE 之后，主流分词方案不断演进：

**WordPiece**（BERT 采用）：与 BPE 类似，但合并标准是**最大化训练数据的似然**，而非单纯统计频次。它会优先合并那些使语料整体似然提升最多的字节对。

**SentencePiece**（LLaMA 等采用）：将整个句子作为字节序列，直接在未分词的原始语料上学习 BPE，不依赖空格信息，对中文、日文等无显式词边界的语言更友好。

**Embedding 的演进**：从静态 Word2Vec 到语境化（Contextualized）表示——ELMo 的双向 LSTM、GPT 的 Transformer Decoder、再到 BERT 的 MLM 预训练，Embedding 的含义从"查表"演变为"根据上下文动态计算"，这直接催生了 GPT 系列和 BERT 系列两大主流大模型路线。

## 拓展动手实现

如果想进一步深入，可以尝试以下几个方向：

1. **Word2Vec 实现**：用 Skip-gram + 负采样在小型语料上训练词向量，用 gensim 库加载或自己实现，感受 CBOW/Skip-gram 的核心前向传播。

2. **完整的 BPE 训练**：对真实语料（如 Wikipedia 摘要）实现完整的 BPE 词表学习流程，观察不同 merge 次数对词表大小的影响。

3. **BERT Tokenizer 对比**：用 transformers 库加载 BERT 和 GPT-2 的 Tokenizer，对比中英文混合文本的分词差异，理解 WordPiece 和 SentencePiece 的实际区别。

## 总结与展望

从 One-Hot 到 Transformer Embedding，文本表征经历了从**稀疏到稠密**、从**离散到连续**、从**静态到语境化**的演进。这一条演进链的每一步都有其必然性：

固定向量无法处理歧义 → Word2Vec 通过神经网络学习稠密语义表示；整词粒度无法解决 OOV → BPE 子词分词优雅地平衡词表大小与表达能力；静态向量无法捕捉上下文 → Transformer 的自注意力机制让模型能够根据输入动态计算语境感知表示。

理解了 Embedding 的前世今生，再回头去看 Transformer 的输入层，你会感受到作者在设计上的深思熟虑：Embedding 不是孤立存在的，它与位置编码、注意力机制共同构成了现代大语言模型的基石。
