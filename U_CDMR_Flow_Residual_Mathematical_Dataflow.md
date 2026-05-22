# U-CDMR-Flow-Residual+ 数学数据流说明

本文档从数据进入系统开始，按数据流顺序说明当前 U-CDMR-Flow-Residual+ 的完整设计。这里不从代码角度解释，而是用数学对象、函数、损失函数和输出约束说明每一步在做什么。

核心目标是生成可用于下游裂缝分割训练的 synthetic image-mask pairs：

```text
same-domain normal image
+ same-domain crack mask
+ mask-gated residual flow
= synthetic crack image + synthetic binary mask
```

系统的核心公式固定为：

```text
I_syn = clip(I_normal + gate(M_syn) * Delta_flow, 0, 1)
Y_syn = M_syn
```

其中：

- `I_normal` 是同域 normal 背景图。
- `M_syn` 是同域 synthetic crack mask。
- `Delta_flow` 是 residual rectified flow 采样得到的裂缝 RGB residual。
- `gate(M_syn)` 是只允许局部残差作用的软 gate。
- `Y_syn` 直接等于输入 mask，不从 synthetic image 里反推。

这个方法不是 FLUX、Stable Diffusion、VAE latent diffusion，也不是整图 RGB diffusion。它只在 mask gate 内生成局部裂缝 residual，背景原则上保持不变。

---

## 1. 全局符号

### 1.1 数据域

数据包含三个 domain：

```text
D = {camera, phone, dphone}
```

每张图像样本记为：

```text
r_i = (I_i, y_i, d_i, v_i, H_i, W_i, a_i)
```

含义：

- `I_i in [0,1]^(H_i x W_i x 3)`：RGB 图像。
- `y_i in {crack, normal, broken}`：标签。
- `d_i in D`：domain。
- `v_i`：video name。
- `H_i, W_i`：原生分辨率。
- `a_i`：crack 标注；normal 没有 crack mask。

当前设计只使用：

```text
y_i in {crack, normal}
```

所有 `broken` 被过滤，不进入 manifest、mask、pseudo-normal、flow training、generation、filter、downstream。

### 1.2 原生分辨率约束

所有图像输出保持原生尺寸：

```text
camera: 640 x 640
phone: 640 x 640
dphone: 1408 x 2560
```

训练时允许 batch 内 padding，但不做全局 resize，也不把 dphone 裁成 640。padding 区域通过 valid mask 从 loss 中排除。

### 1.3 像素范围

图像统一归一化为：

```text
I in [0,1]^(H x W x 3)
```

mask 通常为：

```text
M in {0,1}^(H x W)
```

软 gate 为：

```text
G in [0,1]^(H x W)
```

---

## 2. 总数据流

完整数据流可以写成函数链：

```text
manifest_merged
  -> F_manifest
  -> manifest_filtered
  -> F_split
  -> manifest_splits
  -> F_mask
  -> masks_manifest + mask representations
  -> F_pseudo
  -> pseudo_manifest
  -> train residual flow
  -> residual flow checkpoint
  -> sample/generate mask
  -> sample residual by ODE
  -> compose synthetic image-mask pair
  -> filter synthetic
  -> downstream segmentation training/evaluation
```

每一步都可以形式化为一个函数：

```text
F_manifest : raw manifest -> filtered manifest
F_split    : filtered manifest -> split-aware manifest
F_mask     : crack image annotation -> multi-channel mask condition
F_pseudo   : crack image + crack mask -> pseudo-normal image
F_flow     : residual flow matching trainer
F_maskgen  : mask bank or descriptor mask flow -> M_syn
F_sample   : normal image + M_syn + flow checkpoint -> Delta_flow
F_compose  : I_normal + Delta_flow + M_syn -> (I_syn, Y_syn)
F_filter   : synthetic manifest -> filtered synthetic manifest
F_down     : real/synthetic segmentation dataset -> downstream segmenter
```

---

## 3. Manifest 过滤

### 3.1 输入

输入是合并后的 manifest。每条记录至少包含：

```text
dataset_group
label
dataset_relative_path
annotation_relative_path
image_width
image_height
video_name
```

### 3.2 过滤函数

定义：

```text
F_manifest(r_i) =
  keep, if y_i in {crack, normal}
  discard, otherwise
```

也就是：

```text
R_filtered = { r_i | y_i in {crack, normal} }
```

并写入：

```text
domain_i = dataset_group_i
```

### 3.3 设计目的

这样保证：

- `broken` 不参与任何训练或生成。
- domain 信息统一为 `camera / phone / dphone`。
- 后续 same-domain generation 有明确条件。

---

## 4. Split 防泄漏

### 4.1 Split 单位

split 不是按图片随机分，而是按：

```text
split_key = domain :: video_name
```

也就是同一个 domain 下同一个视频的所有帧进入同一个 split。

### 4.2 Split 函数

对每个 domain：

```text
V_d = unique video names in domain d
```

随机打乱后按比例分配：

```text
train : val : test = 0.70 : 0.15 : 0.15
```

得到：

```text
S(d, v) in {train, val, test}
```

每条样本继承：

```text
split_i = S(d_i, v_i)
```

### 4.3 设计目的

即使当前数据已经去掉相邻帧，也仍然防止同一个视频内的相似背景、光照、拍摄位置同时出现在 train 和 test 中。

下游判断 synthetic 是否有效时，test split 必须保持真实数据，不被 teacher、filter、人眼挑选或 synthetic 生成反向影响。

---

## 5. Crack Mask 表示

### 5.1 原始 mask

对每个 crack 样本，从人工标注得到二值 mask：

```text
M_raw in {0,1}^{H x W}
```

其中：

```text
M_raw(p) = 1, if pixel p belongs to crack
M_raw(p) = 0, otherwise
```

### 5.2 形态学膨胀

定义半径为 `r` 的 dilation：

```text
D_r(M)(p) = max_{q in B_r(p)} M(q)
```

其中 `B_r(p)` 是以像素 `p` 为中心的半径 `r` 邻域。

### 5.3 Mask 区域

当前每个 crack mask 会产生 7 个 condition channel。

#### 5.3.1 Raw mask

```text
M_raw
```

真实裂缝标注区域，也是最终 synthetic label。

#### 5.3.2 Inpaint mask

```text
M_inpaint = D_{r_inpaint}(M_raw)
```

用于 pseudo-normal 擦除裂缝。它比真实裂缝略宽，目的是尽可能覆盖裂缝及其边缘痕迹。

#### 5.3.3 Band mask

```text
M_band = D_{r_band}(M_raw)
```

用于 residual flow 的主要 loss 区域。它比 `M_raw` 宽一点，让模型不只学习标注内部像素，也学习裂缝边缘和过渡。

#### 5.3.4 Gate mask

先做二值 gate：

```text
M_gate_bin = D_{r_gate}(M_raw)
```

再做 Gaussian blur：

```text
M_gate = G_sigma(M_gate_bin)
```

其中 `M_gate in [0,1]`。

这个 gate 控制 residual 真正加到 normal 图上的范围：

```text
I_syn = I_normal + M_gate * Delta_flow
```

如果 `M_gate = 0`，背景完全不变。

#### 5.3.5 Skeleton

对 `M_raw` 做 thinning：

```text
M_skeleton = Thin(M_raw)
```

它近似描述裂缝中心线。

#### 5.3.6 Signed distance field

定义到 mask 的 signed distance：

```text
SDF(p) =
  +dist(p, boundary(M_raw)), if p inside M_raw
  -dist(p, boundary(M_raw)), if p outside M_raw
```

归一化到 `[0,1]`：

```text
M_sdf(p) = clip((SDF(p) + c) / (2c), 0, 1)
```

其中 `c` 是截断距离。

#### 5.3.7 Thickness

在 mask 内估计局部厚度：

```text
M_thickness(p) ~= 2 * dist(p, boundary(M_raw)) / c
```

并裁剪到 `[0,1]`。

### 5.4 Condition tensor

把 7 个 mask channel 堆叠：

```text
C_M = concat(
  M_raw,
  M_inpaint,
  M_band,
  M_gate,
  M_skeleton,
  M_sdf,
  M_thickness
)
```

因此：

```text
C_M in R^{7 x H x W}
```

Residual flow 和 generation 都使用这个 mask condition。

### 5.5 Domain-specific 半径

当前默认：

```text
camera:
  r_inpaint = 9
  r_band    = 5
  r_gate    = 7
  gate_blur = 3

phone:
  r_inpaint = 9
  r_band    = 5
  r_gate    = 7
  gate_blur = 3

dphone:
  r_inpaint = 11
  r_band    = 7
  r_gate    = 9
  gate_blur = 4
```

dphone 分辨率更高，所以半径略大。

---

## 6. Mask 描述子

每个 mask 还会计算一个低维 descriptor，用于统计和 descriptor mask flow。

### 6.1 几何量

设：

```text
A = sum_p M_raw(p)
N = H * W
```

mask 面积比例：

```text
a = A / N
```

mask 中心：

```text
c_x = mean{x | M_raw(x,y)=1} / W
c_y = mean{y | M_raw(x,y)=1} / H
```

bounding box 尺寸：

```text
b_w = (x_max - x_min + 1) / W
b_h = (y_max - y_min + 1) / H
```

主方向用 mask 像素坐标 PCA 得到。令 mask 坐标为：

```text
P = {(x,y) | M_raw(x,y)=1}
```

计算协方差矩阵：

```text
Sigma = cov(P)
```

最大特征值对应特征向量 `u` 给出主方向：

```text
theta = atan2(u_y, u_x)
```

并折叠到：

```text
theta in [-pi/2, pi/2]
```

归一化方向：

```text
theta_norm = (theta + pi/2) / pi
```

### 6.2 Descriptor 向量

当前 descriptor 为 9 维：

```text
z = [
  c_x,
  c_y,
  a,
  b_w,
  b_h,
  theta_norm,
  skeleton_ratio,
  thickness_mean,
  component_count_norm
]
```

其中：

```text
skeleton_ratio = sum(M_skeleton) / (H * W)
component_count_norm = min(component_count, 10) / 10
```

### 6.3 设计目的

这个 descriptor 不是最终 mask label，而是 mask 生成的中间几何空间。它把裂缝的位置、面积、尺度、方向、拓扑粗略编码成一个低维点，方便做 descriptor-level flow matching。

---

## 7. Pseudo-normal 生成

Residual flow 不能直接用 random normal 和 real crack 做监督，因为真实 normal 与真实 crack 不是严格配对图。当前训练采用 self-reconstruction：

```text
real crack image -> erase crack -> pseudo-normal
```

### 7.1 输入

对每个 crack 样本：

```text
I_crack in [0,1]^{H x W x 3}
M_inpaint in {0,1}^{H x W}
M_gate in [0,1]^{H x W}
```

### 7.2 擦除函数

定义 pseudo-normal 生成函数：

```text
I_ctx = E(I_crack, M_inpaint)
```

其中 `E` 可以是：

- OpenCV Telea inpainting；
- OpenCV Navier-Stokes inpainting；
- local blur；
- local median / texture copy。

目标是让：

```text
I_ctx outside M_inpaint ~= I_crack outside M_inpaint
```

并且在 `M_inpaint` 内尽量像没有裂缝的背景。

### 7.3 Pseudo-normal 质量指标

定义灰度差异：

```text
A(p) = gray(|I_crack(p) - I_ctx(p)|)
```

外部变化：

```text
outside_l1 = mean_{p notin M_gate} A(p)
```

内部变化：

```text
inside_l1 = mean_{p in M_inpaint} A(p)
```

gate band artifact：

```text
artifact_band_energy = mean_{p in M_gate and p notin M_inpaint} A(p)
```

质量分数：

```text
pseudo_quality_score =
  max(0, inside_l1 - outside_l1 - 0.5 * artifact_band_energy)
```

接受条件：

```text
outside_l1 <= max_outside_l1
pseudo_quality_score >= min_quality_score
```

### 7.4 设计目的

`I_ctx` 是 residual flow 的条件图。它模拟“没有裂缝但背景一致”的图像。训练目标不是从 random normal 生成某个真实 crack，而是学习：

```text
I_ctx + crack residual -> I_crack
```

这样真实监督是同一张图内部构造出来的，避免 paired normal 不存在的问题。

---

## 8. Residual Flow Matching 训练

这是当前主生成器。

### 8.1 训练样本

一个训练样本包含：

```text
I_ctx      pseudo-normal image
I_crack    original real crack image
C_M        7-channel mask condition
d          domain index
s          style vector
V          valid mask
```

其中 `V` 用来排除 batch padding：

```text
V(p) = 1, if p is real image pixel
V(p) = 0, if p is padding pixel
```

### 8.2 Gate support

定义二值 gate support：

```text
G_support(p) = 1[M_gate(p) > 0] * 1[V(p) > 0.5]
```

它用于限制 residual 只在 gate 内存在。

### 8.3 真实 residual 终点

真实 residual：

```text
R_gt = I_crack - I_ctx
```

flow 目标终点：

```text
x_1 = G_support * R_gt
```

注意 `x_1` 是 RGB residual，不是整张图。

### 8.4 噪声起点

采样：

```text
epsilon ~ N(0, I)
```

定义：

```text
x_0 = sigma * epsilon * G_support
```

当前默认：

```text
sigma = 0.35
```

也就是说噪声只在 gate 内存在。

### 8.5 Rectified flow interpolation

采样时间：

```text
t ~ Uniform(0,1)
```

构造中间状态：

```text
x_t = (1 - t) x_0 + t x_1
```

目标速度场：

```text
v_target = x_1 - x_0
```

因为当前是 straight-line rectified flow，所以速度目标不显式依赖 `t`。

### 8.6 模型函数

Residual flow model 定义为：

```text
v_pred = F_theta(
  x_t,
  t,
  I_ctx,
  C_M,
  d,
  s
)
```

输出：

```text
v_pred in R^{3 x H x W}
```

它预测 RGB residual velocity field。

### 8.7 模型结构设计

模型是 mask-conditioned U-Net velocity model：

```text
input = concat(x_t, I_ctx, C_M)
```

输入通道数：

```text
3 residual channels
+ 3 context RGB channels
+ 7 mask condition channels
= 13 channels
```

时间 `t` 使用 sinusoidal embedding：

```text
gamma_t = SinCosEmbedding(t)
```

domain 使用 embedding：

```text
gamma_d = Embedding(d)
```

style vector 使用线性映射：

```text
gamma_s = Linear(s)
```

三者加到 bottleneck：

```text
gamma = gamma_t + gamma_d + gamma_s
```

因此模型同时知道：

- 当前 flow 时间；
- 当前 domain；
- 当前 mask 几何；
- 当前背景图；
- 一个随机 style 控制向量。

### 8.8 Style vector

训练时：

```text
s ~ N(0, I)
```

并带 dropout：

```text
s = 0 with probability p_style_dropout
```

当前默认：

```text
style_dim = 16
style_dropout = 0.5
```

设计目的：

- 有 style 时允许 residual 有随机变化；
- style 被置零时模型仍能在无随机风格条件下工作。

### 8.9 Loss 权重区域

定义 valid weight：

```text
W_valid = V
```

定义 inside weight：

```text
W_inside = clip(M_band + 0.25 * M_gate, 0, 1) * W_valid
```

定义 outside weight：

```text
W_outside = 1[M_gate <= 0.001] * W_valid
```

`W_inside` 让模型重点学习 crack band 内的速度。`W_outside` 强迫 mask 外速度接近 0。

### 8.10 Flow MSE

```text
L_flow =
  sum_p W_inside(p) * ||v_pred(p) - v_target(p)||_2^2
  /
  max(sum_p W_inside(p), 1)
```

### 8.11 Outside MSE

```text
L_outside =
  sum_p W_outside(p) * ||v_pred(p)||_2^2
  /
  max(sum_p W_outside(p), 1)
```

它惩罚 mask 外预测速度。

### 8.12 Velocity leak L1

```text
L_leak =
  sum_p W_outside(p) * ||v_pred(p)||_1
  /
  max(sum_p W_outside(p), 1)
```

它比 MSE 更直接约束 mask 外任何小幅 residual。

### 8.13 总损失

```text
L_residual_flow =
  lambda_flow    * L_flow
+ lambda_outside * L_outside
+ lambda_leak    * L_leak
```

当前默认：

```text
lambda_flow    = 1.0
lambda_outside = 0.25
lambda_leak    = 1.0
```

### 8.14 训练优化

优化器：

```text
AdamW(theta)
```

默认：

```text
learning_rate = 2e-4
weight_decay  = 1e-4
grad_clip     = 1.0
AMP           = enabled
```

分布式训练通过多 GPU DDP 执行。batch 内不同尺寸图像不 resize，只 padding 到同 batch 最大尺寸，再用 valid mask 排除 padding。

### 8.15 当前训练稳定性问题

当前训练缺少 NaN guard。实际训练中出现过：

```text
epoch 19 / global_step 19560 之后 loss = NaN
```

后续 checkpoint 继续保存，导致 `latest.pt` 被坏 checkpoint 覆盖。因此实验上应把最后干净 checkpoint 显式固定为：

```text
epoch_0018.pt
```

后续需要加入：

```text
if not finite(loss):
  stop training
  do not overwrite latest.pt
  save bad-batch diagnostics
```

---

## 9. Mask 生成路线

生成时 mask 不由 residual flow 生成。Residual flow 只生成 RGB residual。mask 先由同域 mask source 给出。

### 9.1 Mask source A: mask bank

mask bank 直接从训练 split 的真实 crack mask 中采样：

```text
M_syn = M_real_j
```

限制：

```text
domain(M_syn) = domain(I_normal)
shape(M_syn)  = shape(I_normal)
```

这条路线最保守，适合验证 residual renderer/flow 是否能把裂缝外观合成到 normal 背景上。

### 9.2 Mask source B: descriptor mask flow

descriptor mask flow 在低维 descriptor 空间中生成 mask 几何。

#### 9.2.1 训练数据

真实 mask descriptor：

```text
z_1 in [0,1]^9
```

domain one-hot：

```text
e_d in {0,1}^3
```

#### 9.2.2 Descriptor flow 起点

```text
z_0 ~ Uniform(0,1)^9
```

#### 9.2.3 Descriptor interpolation

```text
t ~ Uniform(0,1)
z_t = (1 - t) z_0 + t z_1
u_target = z_1 - z_0
```

#### 9.2.4 Descriptor velocity model

```text
u_pred = H_phi(z_t, t, e_d)
```

其中 `H_phi` 是 MLP。

#### 9.2.5 Descriptor loss

```text
L_descriptor_flow =
  mean ||u_pred - u_target||_2^2
```

#### 9.2.6 Descriptor sampling

从随机 descriptor 出发：

```text
z_0 ~ Uniform(0,1)^9
```

用 Euler ODE：

```text
z_{k+1} = clip(z_k + Delta t * H_phi(z_k, t_k, e_d), 0, 1)
```

得到：

```text
z_syn = z_K
```

#### 9.2.7 Descriptor 到 mask

系统不会直接从 descriptor 画任意自由形状，而是：

1. 找到与 `z_syn` 几何相似的真实 template mask；
2. 裁出 template 的 bbox；
3. 按 `bbox_w, bbox_h` resize；
4. 按 `theta` rotate；
5. 按 `center_x, center_y` 放置回原 canvas；
6. 对新 mask 重新构建 7-channel condition。

因此 descriptor flow 生成的是：

```text
mask geometry + placement variation
```

而不是 residual appearance。

### 9.3 当前主路线与检查路线

方法目标的主路线是：

```text
mask_source = descriptor_flow
residual_source = flow
```

但如果 descriptor-flow mask artifacts 尚未准备好，可以先用：

```text
mask_source = bank
residual_source = flow
```

这条路线用于检查 residual flow 是否能生成合理裂缝外观。

---

## 10. Residual Flow 推理 / ODE Sampling

### 10.1 输入

推理时输入：

```text
I_normal
C_M_syn
d
s
checkpoint theta
```

其中：

```text
C_M_syn = condition(M_syn)
```

### 10.2 初始噪声

```text
x_0 = sigma * epsilon * 1[M_gate > 0]
epsilon ~ N(0,I)
```

默认：

```text
sigma = 0.35
```

### 10.3 Euler sampler

令：

```text
Delta t = 1 / K
t_k = k / K
```

Euler：

```text
v_k = F_theta(x_k, t_k, I_normal, C_M_syn, d, s)
x_{k+1} = (x_k + Delta t * v_k) * 1[M_gate > 0]
```

### 10.4 Heun sampler

当前默认使用 Heun：

```text
v_k = F_theta(x_k, t_k, I_normal, C_M_syn, d, s)
tilde{x}_{k+1} = (x_k + Delta t * v_k) * 1[M_gate > 0]
```

再计算：

```text
tilde{v}_{k+1} =
  F_theta(tilde{x}_{k+1}, t_k + Delta t, I_normal, C_M_syn, d, s)
```

最终：

```text
x_{k+1} =
  (x_k + 0.5 * Delta t * (v_k + tilde{v}_{k+1}))
  * 1[M_gate > 0]
```

### 10.5 Delta clamp

ODE 结束得到：

```text
Delta_flow = x_K
```

为了防止过强残差，做裁剪：

```text
Delta_flow = clip(Delta_flow, -delta_max, delta_max)
```

当前默认：

```text
delta_max = 0.5
```

### 10.6 合成图像

最终合成：

```text
I_syn = clip(I_normal + M_gate * Delta_flow, 0, 1)
Y_syn = M_raw
```

这里 `M_gate` 是软融合区域，`M_raw` 是最终二值 label。

### 10.7 为什么 mask 外背景不变

因为每一步 ODE 后都会执行：

```text
x_k = x_k * 1[M_gate > 0]
```

并且最终合成也是：

```text
I_syn = I_normal + M_gate * Delta_flow
```

所以在：

```text
M_gate(p) = 0
```

的像素上：

```text
I_syn(p) = I_normal(p)
```

这也是 synthetic image-mask pair 能用于分割训练的关键：mask 外不应被生成器乱改。

---

## 11. Synthetic 输出与质量指标

每个 synthetic sample 输出：

```text
synthetic image: I_syn
synthetic mask:  Y_syn = M_raw
residual_abs:    gray(|M_gate * Delta_flow|)
metadata row
```

### 11.1 Change map

```text
Change(p) = |I_syn(p) - I_normal(p)|
```

### 11.2 Inside change

```text
inside_change =
  mean_{p in M_raw} mean_c Change_c(p)
```

表示真实 label 区域内部平均改变强度。

### 11.3 Outside change

```text
outside_change =
  mean_{p: M_gate(p) <= 0.001} mean_c Change_c(p)
```

表示 gate 外背景是否被改变。

### 11.4 Residual leakage

```text
residual_leak =
  outside_change / max(inside_change, 1e-8)
```

越低越好。

### 11.5 Residual support

先定义 residual support：

```text
S_res(p) = 1[gray(Change(p) * 255) > tau]
```

默认阈值：

```text
tau = 8
```

### 11.6 Mask-residual IoU

```text
mask_residual_iou =
  |S_res intersect M_raw| / |S_res union M_raw|
```

它衡量实际发生变化的位置是否与 label mask 对齐。

### 11.7 Teacher metrics

如果加载已有 teacher segmentation model，则：

```text
P_teacher = Teacher(I_syn)
Y_teacher = 1[P_teacher >= threshold]
```

计算：

```text
teacher_dice
teacher_recall_on_mask
teacher_fp_outside_mask
```

这些不是最终裁判，只是 synthetic 质量辅助信号。

### 11.8 Metadata

每条 synthetic manifest 至少记录：

```text
synthetic_image_path
synthetic_mask_path
source_normal_path
domain
mask_source
residual_source = flow
residual_flow_checkpoint
mask_flow_checkpoint
flow_steps
flow_sampler
flow_sigma
flow_max_delta
seed_residual
seed_mask
mask_area_ratio
outside_change
inside_change
residual_leakage_score
mask_residual_iou
generation_formula
```

这样可以追踪每张 synthetic image-mask pair 是如何生成的。

---

## 12. Synthetic Filtering

Filter 不改变图像和 mask，只给 synthetic sample 打分并决定是否保留。

### 12.1 Domain-specific thresholds

每个 domain 可以有自己的阈值：

```text
T_d = {
  max_residual_leak,
  max_outside_change,
  min_mask_area,
  max_mask_area,
  min_mask_residual_iou,
  min_teacher_dice,
  min_teacher_recall
}
```

默认三域当前相同，但接口支持 per-domain。

### 12.2 Keep 条件

对 synthetic 样本 `j`：

```text
keep_j =
  residual_leak_j <= max_residual_leak_d
  and outside_change_j <= max_outside_change_d
  and min_mask_area_d <= mask_area_ratio_j <= max_mask_area_d
  and mask_residual_iou_j >= min_mask_residual_iou_d
```

如果 teacher 阈值存在并且该样本有 teacher 分数，再加：

```text
teacher_dice_j >= min_teacher_dice_d
teacher_recall_j >= min_teacher_recall_d
```

### 12.3 Quality score

当前分数形式为：

```text
score =
  1
  - 0.3 * min(residual_leak / max_residual_leak, 2)
  - 0.3 * min(outside_change / max_outside_change, 2)
  + 0.4 * min(mask_residual_iou, 1)
```

如果 teacher 分数存在：

```text
score += 0.1 * clamp(teacher_dice, 0, 1)
score += 0.1 * clamp(teacher_recall, 0, 1)
```

如果 topology score 存在：

```text
score += 0.1 * clamp(topology_score, 0, 1)
```

---

## 13. 下游分割训练

最终目标不是生成图看起来好，而是提升 held-out real test 上的分割性能。

### 13.1 训练数据

下游 segmentation dataset 可以包含：

```text
real crack images
real normal images
filtered synthetic crack images
```

真实 crack：

```text
Y_real = human annotation mask
```

真实 normal：

```text
Y_real = all-zero mask
```

synthetic：

```text
Y_syn = M_syn
```

### 13.2 Segmentation model

下游模型是 encoder-decoder segmenter，常用 encoder：

```text
ResNet34 pretrained encoder
```

输出：

```text
logits = U_psi(I)
P = sigmoid(logits)
```

### 13.3 Valid mask

与 residual flow 一样，不同尺寸 batch 内 padding，padding 区域由：

```text
V(p)
```

排除。

### 13.4 BCE loss

带正样本权重：

```text
L_bce =
  sum_p V(p) * BCEWithLogits(logits(p), Y(p); pos_weight)
  /
  max(sum_p V(p), 1)
```

默认：

```text
pos_weight = 8.0
```

### 13.5 Dice loss

```text
P = sigmoid(logits)
```

```text
Dice =
  (2 * sum_p P(p) Y(p) V(p) + eps)
  /
  (sum_p (P(p) + Y(p)) V(p) + eps)
```

```text
L_dice = 1 - Dice
```

### 13.6 Focal loss

定义：

```text
p_t = P * Y + (1 - P) * (1 - Y)
```

```text
L_focal =
  sum_p V(p) * (-(1 - p_t)^2 * log(clamp(p_t, 1e-6, 1)))
  /
  max(sum_p V(p), 1)
```

### 13.7 总分割损失

```text
L_seg =
  w_bce   * L_bce
+ w_dice  * L_dice
+ w_focal * L_focal
```

默认：

```text
w_bce   = 1.0
w_dice  = 1.0
w_focal = 0.5
```

### 13.8 对照实验

最终应比较：

```text
A. real only
B. real + mask bank + residual flow synthetic
C. real + descriptor mask flow + residual flow synthetic
```

如果后面保留旧 deterministic renderer，则只能作为 baseline，不是主方法。

---

## 14. 下游评估指标

对预测：

```text
Y_pred = 1[sigmoid(logits) >= threshold]
```

和真实 mask：

```text
Y
```

定义：

```text
TP = |Y_pred intersect Y|
FP = |Y_pred intersect not Y|
FN = |not Y_pred intersect Y|
```

### 14.1 Dice

```text
Dice = 2TP / (2TP + FP + FN)
```

### 14.2 IoU

```text
IoU = TP / (TP + FP + FN)
```

### 14.3 Precision

```text
Precision = TP / (TP + FP)
```

### 14.4 Recall

```text
Recall = TP / (TP + FN)
```

### 14.5 Boundary F1

定义边界：

```text
B(M) = M - erode(M)
```

允许 tolerance 半径 `r`：

```text
Precision_boundary =
  |B(Y_pred) intersect dilate(B(Y), r)| / |B(Y_pred)|
```

```text
Recall_boundary =
  |B(Y) intersect dilate(B(Y_pred), r)| / |B(Y)|
```

```text
BoundaryF1 =
  2 * Precision_boundary * Recall_boundary
  /
  (Precision_boundary + Recall_boundary)
```

### 14.6 Normal false positive rate

当真实 mask 为空：

```text
NormalFPR = |Y_pred| / (H * W)
```

### 14.7 Size buckets

根据 mask area ratio：

```text
area_ratio = |Y| / (H * W)
```

分桶：

```text
normal: area = 0
tiny:   area_ratio < 1e-5
small:  area_ratio < 1e-4
medium: area_ratio < 1e-3
large:  otherwise
```

最终必须按 domain 和 size bucket 报告。

---

## 15. 实验日志与诊断设计

### 15.1 当前已有日志

当前已有的实验记录包括：

```text
per-step stdout:
  loss
  flow_mse
  outside_mse
  velocity_leak

per-epoch json:
  epoch metrics
  args
  seconds

checkpoint:
  model state
  optimizer state
  epoch
  global_step
  args
  metrics

generation manifest:
  residual checkpoint
  mask source
  flow steps
  sampler
  sigma
  seed
  quality metrics
```

这些能看出 loss 是否下降，也能追踪每张 synthetic 图怎么来的。

### 15.2 当前不足

目前不足以完整解释 flow 为什么坏：

```text
No per-step JSONL/CSV curve
No t-bin loss
No per-domain residual-flow loss
No x_t / v_pred / v_target visualization
No ODE trajectory visualization
No NaN guard
No clean latest checkpoint protection
```

这次训练 NaN 后仍继续跑，并覆盖 `latest.pt`，说明必须补稳定性日志。

### 15.3 建议补充的 residual-flow 日志

每一步记录：

```text
global_step
epoch
loss
flow_mse
outside_mse
velocity_leak
learning_rate
grad_norm
t_mean
t_min
t_max
x0_norm
x1_norm
xt_norm
target_v_norm
pred_v_norm
delta_norm
nan_flag
domain_histogram
```

### 15.4 t-bin loss

把时间分桶：

```text
[0.0,0.2), [0.2,0.4), [0.4,0.6), [0.6,0.8), [0.8,1.0]
```

记录：

```text
L_flow_bin(k) =
  mean L_flow over samples with t in bin k
```

这能判断模型是在靠近 noise 端不稳定，还是靠近 target residual 端不稳定。

### 15.5 向量场可视化

每隔固定 step 保存：

```text
I_ctx
I_crack
M_raw
M_gate
x_t_abs
target_delta_abs = |x_1|
pred_velocity_abs = |v_pred|
target_velocity_abs = |v_target|
error_abs = |v_pred - v_target|
preview_recon = clip(I_ctx + M_gate * x_t, 0, 1)
```

如果要看方向场，可以在 mask 附近稀疏采样像素，画：

```text
arrow(p) = (v_R(p), v_G(p), v_B(p))
```

但 RGB residual 的 velocity 不是二维几何运动场，所以更推荐看 residual magnitude map 和 RGB residual image，而不是传统 optical-flow 箭头。

### 15.6 ODE 轨迹可视化

生成时保存：

```text
x_0
x_4
x_8
x_16
x_24
x_32
```

以及：

```text
I_k = clip(I_normal + M_gate * x_k, 0, 1)
```

这样能看裂缝 residual 是否逐步形成。

### 15.7 NaN guard

训练中应加入：

```text
if not isfinite(loss):
  save bad batch ids
  save t, domain, mask stats
  save x0/x1/xt norms
  stop or skip according to policy
  do not update optimizer
  do not overwrite latest.pt
```

checkpoint 规则应改为：

```text
latest_clean.pt only if all metrics finite
latest.pt optional
bad_nan_epoch_xxxx.pt separate
```

---

## 16. 当前一次推理结果如何理解

最近一次 inspection 使用：

```text
residual checkpoint: epoch_0018.pt
mask_source: bank
residual_source: flow
flow_steps: 32
flow_sampler: heun
flow_sigma: 0.35
max samples: 12
```

输出说明：

```text
normal       原始 normal 背景
synthetic    I_normal + gate(M) * Delta_flow
mask         Y_syn = M_raw
residual_abs |gate(M) * Delta_flow|
```

观察结果：

- mask 外 residual leak 接近 0，说明 gate 约束有效。
- 背景基本保持不变。
- 但裂缝外观偏黑斑/补丁，不像细长裂纹。
- 这说明当前 residual flow 学到了“局部改变”，但 appearance quality 还不足。

因此下一步不是直接下游训练，而是先修：

```text
NaN guard
trajectory visualization
t-bin/domain-bin diagnostics
descriptor-flow mask generation
residual appearance quality
```

---

## 17. 整体方法的数学总结

### 17.1 Training

对真实 crack：

```text
I_ctx = E(I_crack, M_inpaint)
x_1 = 1[M_gate > 0] * (I_crack - I_ctx)
x_0 = sigma * epsilon * 1[M_gate > 0]
x_t = (1 - t)x_0 + t x_1
v_target = x_1 - x_0
v_pred = F_theta(x_t, t, I_ctx, C_M, d, s)
```

优化：

```text
min_theta L_residual_flow
```

### 17.2 Generation

对 same-domain normal：

```text
I_normal ~ normal images in domain d
M_syn ~ same-domain mask bank or descriptor mask flow
C_M_syn = condition(M_syn)
x_0 = sigma * epsilon * 1[M_gate > 0]
Delta_flow = ODESample(F_theta, x_0, I_normal, C_M_syn, d, s)
I_syn = clip(I_normal + M_gate * Delta_flow, 0, 1)
Y_syn = M_raw
```

### 17.3 Final success criterion

最终成功不是看生成图“像不像”，而是：

```text
real-only downstream segmenter
vs
real + flow synthetic downstream segmenter
```

在真实 held-out test 上是否提升：

```text
Dice
IoU
Recall
Boundary F1
tiny/small crack recall
normal false positive rate
per-domain metrics
```

并且不能提高 normal false positives。

