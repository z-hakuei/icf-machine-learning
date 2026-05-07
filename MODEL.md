# 模型说明：任意次序自回归 Transformer + GMM

本仓库实现了 PPT《机器学习横版.pdf》第 4 页所描述的统一模型，
并在 `combined_data.xlsx`（5 个 input + 15 个 output，共 10000 个样本）上完成训练与评估。

---

## 1. 任务定位

PPT 把 ICF 的设计问题归纳为三类：

| 任务 | 数学形式 | 在我们的数据里 |
| --- | --- | --- |
| **代理模型**（正向回归） | `p(y \| x)` | 给定 5 个 input → 预测 15 个 output |
| **辅助设计**（逆问题 / 约束优化） | `p(x \| y)` | 给定 output 测量值 → 给出可行的 input 后验 |
| **定标率**（输出之间的关联） | `p(y_2 \| y_1)` | 在 output 内部找到关联曲线 |

PPT 给出的核心想法：**只要训练一个能拟合任意条件分布
`p(x_S | x_{S^c})` 的模型，上述 3 个任务就由同一个网络一并解决**。
我们完全照搬这套思路。

---

## 2. 数学原理

记 20 维联合变量为 `x = (x_1, ..., x_{20})`。任何高维分布都可写成
条件分布的乘积（链式法则）：

```
p(x_1, x_2, ..., x_n) = p(x_{σ(1)}) · p(x_{σ(2)} | x_{σ(1)}) · ... · p(x_{σ(n)} | x_{σ(1)},...,x_{σ(n-1)})
```

其中 `σ` 是任意排列。模型不必把 `σ` 固定下来——只要它能给出**任意子集的条件**
`p(x_S | x_{S^c})`，就等价于学到了完整的联合分布
（Uria 2014 的 NADE / ARDM 思想）。

训练时我们随机选一个掩码集合 `S`、隐藏 `x_S`、令网络从 `x_{S^c}` 预测它们的密度。
按 `r ∼ U(0,1)` 抽样掩码比例并 KL 等价于 `D_KL(p_data ‖ p_model)`
（Goodfellow 2016 / PPT 第 4 页右下角的注释）。

---

## 3. 网络结构

PPT 第 4 页图右侧的方框逐字落到代码里 (`model.py`):

| 层 | 形状 | 备注 |
| --- | --- | --- |
| 每个 token 的 raw 特征 | `[value, mask, onehot_id]` (1+1+20=22) | `mask=1` 表示该维被隐藏 |
| `Linear[22, d_model]` | `d_model=64` | 把 token 嵌入到 64 维 |
| `Self-attention layers × 4` | 4 头 / `d_ff=256` | 全双向注意力，无因果掩码 |
| `Linear[d_model, 3K]` | `K=10` | 输出每维 GMM 的 (logits w, μ, log σ) |
| **GMM 头** | `K=10` 高斯 | 显式输出概率密度 |

总参数：**0.20 M**（CPU 训练 120 轮 ≈ 200 s；如果改回 PPT 的 `d_model=128, layers=8, K=20` 则 1.6 M 参数）。

> **注**：PPT 用 `序列长度=28`、`K_GMM=20`、`d_model=128`。我们这里因为
> 数据维度 = 20、CPU 训练，把规模适度缩小，但**架构、损失、采样流程完全一致**。

---

## 4. 训练流程 (`train.py`)

1. 读 `combined_data.xlsx`，把所有列做 `(x − μ) / σ` 标准化（保存 `μ, σ` 用于反归一化）。
2. 9000 训练 / 1000 测试随机划分（种子 0），保存 `test_indices.npy`。
3. 每个 batch：
   - 抽 `r ∼ U(1/N, 1−1/N)`；
   - 每维以概率 `r` 被掩；保证每条样本至少 1 个被掩位；
   - 计算
     `L = − E_{S} (1/|S|) Σ_{i∈S} log p_θ(x_i | x_{S^c})`
     即 PPT 所说的 “masked-loglik / |masked|”。
4. AdamW (`lr=2e-3`, `wd=1e-5`) + Cosine schedule，120 epochs。
5. 训练 / 测试 NLL、正向 / 逆向 RMSE 每轮记录到 `artifacts/history.json`。

---

## 5. 推理三种模式 (`evaluate.py`)

### 5.1 正向代理模型
把 5 个 input 全部置为可见（`mask=0`），15 个 output 置为隐藏（`mask=1`），
取网络输出的 GMM 均值 `Σ w_k μ_k` 作为点估计。

### 5.2 逆向代理模型
把 15 个 output 置为可见、5 个 input 置为隐藏，取均值即可。
**注意**：逆问题往往是一对多，单点均值会"塌缩到中位"，所以 R² 会低。
真正可信的是 5.3 节的后验分布。

### 5.3 任意条件下的后验分布（PPT 第 8、10 页）
- 一维：直接读出 `p(x_i | conditions)` 的 GMM 解析式。
- 多维：按链式法则**自回归采样**。代码里 `model.sample_joint(...)` 就是 PPT
  第 11 页 "采样 → 模型 → 采样 → 模型 → ..." 流程的实现。

---

## 6. 文件清单

```
combined_data.xlsx          - 输入数据（外部提供）
model.py                    - AnyOrderARTransformer + GMMHead
train.py                    - 训练脚本
evaluate.py                 - 评估 + 画图脚本
report.tex                  - LaTeX 报告
report.pdf                  - 编译后的 PDF（如本机有 LaTeX）
MODEL.md                    - 本文件
artifacts/
    model.pt                - 训练好的权重 + 配置 + 归一化系数
    history.json            - 每轮 loss / RMSE
    metrics.json            - 最终的 R² / RMSE（每维）
    test_indices.npy        - 测试集索引
    figs/01..06_*.png       - 报告里用到的所有插图
```

---

## 7. 复现命令

```bash
python train.py        # 约 3.5 分钟（CPU, 120 epochs）
python evaluate.py     # 几秒
tectonic report.tex    # 编译 PDF（也可用 pdflatex / xelatex）
```

---

## 8. 主要结果（与 `combined_data.xlsx` 对应）

- **正向代理**：测试集平均 **R² ≈ 0.944, RMSE ≈ 0.21**（标准化单位）。
- **逆向代理**：测试集平均 **R² ≈ 0.40, RMSE ≈ 0.61**（点估计；逆问题一对多，
  这正是 PPT 强调"看后验而不是点估计"的原因）。
- **学到的边缘分布**：与数据直方图几乎重合（`02_marginals.png`），
  对应 PPT 第 6 页左侧的"模型可以有效拟合数据的分布"。
- **后验**：给定一个测试样本的 outputs，对 inputs 的后验分布
  把真值 (红星) 包在高密区里，并自动呈现某些维度的多模态结构
  （`05_posterior_corner.png`），对应 PPT 第 8、10 页。
