# U-CDMR-Flow-Residual+ 方案设计文档

> 目标：基于真实 `normal` 图像生成可靠的裂缝 `image-mask pairs`，用于提升下游 U-Net 裂缝分割任务。  
> 原则：**同域生成、mask-first、mask-gated residual、不生成整张 RGB、下游闭环筛选、完全忽略 broken。**

---

## 0. 一句话总方案

当前项目不要再走 Stable Diffusion / FLUX 直接 inpainting 整图，也不要从零训练一个大 DiT 直接生成 RGB。推荐主线改成：

```text
same-domain normal image
+ domain-conditioned crack mask
+ mask-gated crack residual
= synthetic crack image + synthetic mask
```

核心公式：

```text
I_syn = I_normal + gate(M_syn) * Delta_crack
Y_syn = M_syn
```

其中：

```text
I_normal      # 真实 normal 背景图
M_syn         # 生成或采样得到的裂缝 mask
Delta_crack   # 只在裂缝区域附近生效的 RGB residual
Y_syn         # 下游 U-Net 使用的 label，直接等于 M_syn
```

最终判断标准不是生成图是否好看，而是：

```text
real + filtered synthetic 训练的 U-Net
是否在真实 held-out crack / normal 上提升 Dice、Recall、Boundary F1，
并且 normal false positive 不上升。
```

---

## 1. 数据使用原则

### 1.1 只使用 crack 和 normal

本方案彻底忽略 broken：

```text
use:    label == crack
use:    label == normal
ignore: label == broken
```

broken 不进入：

```text
生成模型训练
mask prior 训练
residual renderer 训练
teacher U-Net 训练
downstream U-Net 训练
hard negative mining
final evaluation
```

理由很简单：broken 数据构建质量差，语义不稳定，强行使用会污染裂缝分割边界。

### 1.2 有效数据统计

按当前 `manifest_merged.csv`，有效数据为：

```text
crack:  463
normal: 343
```

按 domain：

```text
camera: 157 crack, 38 normal
phone:  106 crack, 36 normal
dphone: 200 crack, 269 normal
```

图像尺寸：

```text
camera: 640x640 + 1920x1080
phone:  640x640 + 1920x1080
dphone: 1408x2560
```

裂缝像素占比极小：

```text
median: 0.3499%
p90:    2.3773%
max:    5.3510%
```

这意味着生成模型必须是：

```text
domain-aware
native-resolution-aware
tiny-crack-aware
mask-alignment-aware
```

### 1.3 同域生成是默认规则

默认不做跨域 RGB 生成：

```text
camera normal -> camera crack synthetic
phone normal  -> phone crack synthetic
dphone normal -> dphone crack synthetic
```

禁止默认做：

```text
camera normal + phone crack style
phone normal  + dphone crack style
dphone normal + camera crack style
```

允许后期只做 mask topology 级别的弱共享，例如：

```text
phone-like thin mask topology + dphone normal + dphone residual renderer
```

但第一版不要做跨域图像风格迁移。

---

## 2. 方法命名

推荐方法名：

```text
U-CDMR-Flow-Residual+
```

全称：

```text
Unified Counterfactual Domain-Calibrated Mask-Residual Flow for Crack Pair Synthesis
```

中文解释：

```text
面向裂缝分割的数据增强方法：
用同域 normal 图作为反事实背景，
生成 domain-conditioned 裂缝 mask，
再生成 mask-gated crack residual，
最终得到天然对齐的 image-mask pair。
```

不建议把主方法叫成单纯的 `Diffusion Baseline` 或 `FLUX Inpainting`，因为你的贡献点不是 text-to-image，而是：

```text
mask-first residual synthesis
same-domain calibration
flow-based mask/residual stochastic generation
segmentation-aware filtering
```

---

## 3. 总体 pipeline

完整流程：

```text
1. 读取 manifest_merged.csv
2. 过滤 broken，只保留 crack / normal
3. LabelMe polygon -> binary mask
4. 统计每个 domain 的 mask area / length / thickness / component 分布
5. 从真实 crack 图构建 pseudo-normal
6. 训练 real-only Teacher Segmenter
7. 训练 Mask Generator
8. 训练 Residual Renderer
9. 用同域 normal 背景生成 synthetic image-mask pairs
10. 用质量过滤器筛 synthetic
11. 用 real + filtered synthetic 训练 downstream U-Net
12. 在真实 held-out crack / normal 上评估
```

核心数据流：

```text
real crack image I_crack + real mask M_real
  -> erase / inpaint crack region
  -> pseudo-normal I_pseudo

I_pseudo + M_real + domain
  -> Residual Renderer
  -> Delta_crack

same-domain I_normal + M_syn + Delta_crack
  -> I_syn = I_normal + gate(M_syn) * Delta_crack
  -> Y_syn = M_syn
```

---

## 4. 模块划分

推荐拆成 6 个模块：

```text
A. Dataset & Mask Parser
B. Pseudo-Normal Builder
C. Teacher Segmenter
D. Mask Generator
E. Residual Renderer
F. Synthetic Filter + Downstream Evaluation
```

---

## 5. A. Dataset & Mask Parser

### 5.1 输入

```text
manifest_merged.csv
LabelMe JSON annotations
image files
```

必须读取字段：

```text
dataset_group            # camera / phone / dphone
label                    # crack / normal / broken
dataset_relative_path
annotation_relative_path
image_width
image_height
crack_pixel_ratio
```

### 5.2 数据过滤

```python
valid_df = df[df["label"].isin(["crack", "normal"])]
```

不要保留 broken 的路径、mask、metadata。

### 5.3 mask rasterization

对 crack 图像：

```text
LabelMe polygon -> binary mask M_raw
```

同时保存：

```text
M_raw              # 原始 label，用于 U-Net 标签
M_skeleton         # skeleton，用于 topology 统计
M_sdf              # signed / unsigned distance field，用于 mask flow
M_thickness        # 裂缝宽度估计
M_inpaint          # dilated mask，用于 pseudo-normal inpainting
M_gate             # dilated + blurred mask，用于 residual blending
```

注意：

```text
M_raw 是最终 label，不能被 blur。
M_inpaint / M_gate 可以 dilate 和 blur。
```

### 5.4 per-domain 统计

每个 domain 单独统计：

```text
mask_area_ratio
bbox_w, bbox_h
skeleton_length
component_count
main_orientation
thickness_mean
thickness_p90
branch_count
center_x, center_y
```

输出：

```text
runs/u_cdmr_flow_residual_plus/stats/domain_mask_stats.csv
runs/u_cdmr_flow_residual_plus/stats/domain_mask_histograms.json
```

这些统计用于：

```text
Mask Generator 的 condition prior
Synthetic Filter 的 domain-specific threshold
Downstream ablation 的 crack-size bucket
```

---

## 6. B. Pseudo-Normal Builder

### 6.1 为什么必须做 pseudo-normal

不能训练：

```text
random normal image + real mask -> real crack image
```

因为 real crack 和 random normal 不是配对图，背景纹理、亮度、位置都不一致。这样训练会让 renderer 学到错误的背景差异。

正确做法：

```text
real crack image + real mask
  -> erase crack region
  -> pseudo-normal image
```

得到真实对齐三元组：

```text
I_pseudo, I_crack, M_real
```

训练 residual renderer 时：

```text
target residual = I_crack - I_pseudo
```

### 6.2 inpainting 方法

第一版可以使用简单稳定的图像修复：

```text
OpenCV Telea / Navier-Stokes
LaMa / Big-LaMa
局部 blur-fill + texture copy
```

推荐顺序：

```text
V1: OpenCV inpaint，快速跑通
V2: LaMa，提升 pseudo-normal 质量
V3: 多方法 ensemble，保留质量最高版本
```

### 6.3 pseudo-normal 质量过滤

过滤掉：

```text
裂缝没有被擦干净
mask 外背景变化过大
inpaint 区域出现明显糊块
边缘出现异常颜色
```

质量指标：

```text
outside_l1 = L1(I_pseudo * (1 - M_inpaint), I_crack * (1 - M_inpaint))
inside_texture_score
teacher_crack_score_on_pseudo
```

如果 teacher 在 pseudo-normal 上仍然强烈预测裂缝，说明擦除不干净，丢弃。

输出：

```text
runs/u_cdmr_flow_residual_plus/pseudo_normal/pseudo_manifest.csv
```

字段：

```text
pseudo_image_path
real_crack_image_path
real_mask_path
domain
image_width
image_height
pseudo_quality_score
outside_l1
inpaint_method
```

---

## 7. C. Teacher Segmenter

### 7.1 作用

Teacher 不是最终模型，而是质量控制器：

```text
1. 过滤 synthetic image-mask pairs
2. 挖掘 normal 图上的 false positive 区域
3. 提供 downstream baseline
4. 帮助判断 synthetic 是否真的有裂缝可见性
```

### 7.2 训练数据

只用真实数据：

```text
real crack  -> real mask
real normal -> empty mask
```

不使用 broken。

### 7.3 模型建议

不要只用随机初始化 CNN U-Net。推荐：

```text
SegFormer / ConvNeXt / DINOv2 encoder + U-Net-like decoder
```

如果短期实现困难，第一版可以用：

```text
ResNet34-UNet
EfficientNet-B3-UNet
ConvNeXt-Tiny-UNet
```

原因：真实数据太少，teacher 如果太弱，后续 synthetic filter 会不可靠。

### 7.4 loss

裂缝极小，不能只用 BCE：

```text
L_teacher = BCEWithLogits + DiceLoss + FocalLoss + BoundaryLoss
```

建议：

```text
BCE weight:       1.0
Dice weight:      1.0
Focal weight:     0.5
Boundary weight:  0.2
```

### 7.5 tile 训练

对高分辨率 dphone：

```text
train: tile crop 768 / 1024
infer: sliding window on full image
save: full-resolution prediction
```

不要把 dphone 全局 resize 到 640，否则细裂缝会被压没。

---

## 8. D. Mask Generator

Mask Generator 是整个方法里最重要的控制模块。原则：

```text
先保证 mask 真实，再让 residual 去贴合 mask。
不要让图像生成模型自己猜裂缝位置。
```

### 8.1 V1：Mask Bank + Descriptor Flow

第一版最稳：

```text
真实 mask template bank
+ descriptor-level flow
+ resize / rotate / place
```

descriptor：

```text
center_x
center_y
area_ratio
bbox_w
bbox_h
angle
length
thickness
component_count
```

flow matching 训练：

```text
x0 ~ N(0, I)
x1 = real_mask_descriptor
t  ~ Uniform(0, 1)
x_t = (1 - t) * x0 + t * x1
v_target = x1 - x0
MLP(x_t, t, domain_embedding) -> v_pred
L_desc_fm = MSE(v_pred, v_target)
```

采样：

```text
sample descriptor
-> select nearest mask template within same domain or same ratio bucket
-> affine transform template
-> paste to native canvas
```

优点：

```text
稳定
不会生成 blob
mask 形态真实
实现快
```

缺点：

```text
多样性有限
本质仍然依赖真实 template bank
```

### 8.2 V2：Latent Dense Mask Flow

第二版增强多样性：

```text
real local mask crop
-> mask autoencoder
-> z_mask
-> latent flow / latent DiT
-> decoded local mask
-> paste back to native canvas
```

mask representation 不只用 binary：

```text
mask logits
skeleton map
SDF map
thickness map
```

训练目标：

```text
x1_mask = z_mask
x0_mask ~ N(0, I)
x_t = (1 - t) * x0_mask + t * x1_mask
v_target = x1_mask - x0_mask
MaskLatentFlow(x_t, t, domain, area_bucket, thickness_bucket) -> v_pred
```

loss：

```text
L_mask_fm
L_mask_bce
L_mask_dice
L_skeleton
L_thickness
L_connectedness
```

### 8.3 推荐实施顺序

```text
Phase 1: mask bank only
Phase 2: descriptor flow + mask bank
Phase 3: latent dense mask flow
```

论文主实验可以最终使用：

```text
descriptor flow as stable baseline
latent dense mask flow as final method
```

---

## 9. E. Residual Renderer

### 9.1 核心原则

Residual Renderer 不生成整张图，只生成局部裂缝 residual：

```text
Delta_crack = Renderer(I_ctx_or_normal, M, domain, style_noise)
I_syn = I_normal + gate(M) * Delta_crack
```

这样可以保证：

```text
背景不乱改
mask 和图像天然对齐
裂缝 label 可直接使用
适合下游分割
```

### 9.2 V1：Pretrained Encoder + Residual Decoder

第一版不要全随机初始化。建议：

```text
f_bg = PretrainedEncoder(I_context)
Delta = ResidualDecoder(f_bg, M_gate, domain_embedding, style_noise)
```

可选 encoder：

```text
ConvNeXt-Tiny
DINOv2-Small
SegFormer-B0/B1 encoder
ResNet34 / ResNet50
```

输入：

```text
I_context       # pseudo-normal or real normal tile
M_raw
M_gate
M_skeleton
M_sdf
M_thickness
domain_id
style_noise
```

输出：

```text
Delta_rgb
```

合成：

```text
I_recon = I_pseudo + gate(M_real) * Delta_rgb
```

训练 target：

```text
I_crack
```

loss：

```text
L_inside_l1       = L1(I_recon * M_band, I_crack * M_band)
L_outside_identity = L1(I_recon * (1 - M_gate), I_pseudo * (1 - M_gate))
L_res_leak        = L1(abs(Delta_rgb) * (1 - M_gate), 0)
L_edge            = edge / gradient loss inside M_band
L_teacher         = Dice(Teacher(I_recon), M_real)    # optional, no gradient in V1 filter-only
```

建议权重：

```text
L_inside_l1:        1.0
L_outside_identity: 2.0
L_res_leak:         2.0
L_edge:             0.3
L_teacher:          0.0 in V1, 0.2 in V2
```

### 9.3 style noise 防止平均化裂缝

Deterministic UNet 很容易生成平均灰黑细线。必须加入 style noise：

```text
z_style ~ N(0, I)
```

控制：

```text
裂缝深浅
边缘锐度
局部高光破坏
黑/白裂纹比例
粗糙纹理
```

训练时做 style dropout：

```text
50% use random z_style
50% use zero z_style
```

生成时每个 mask 采样多个 residual：

```text
same normal + same mask + different z_style -> K candidates
```

再用 filter 选最好的。

### 9.4 V2：Residual Flow Matching

当 V1 可以稳定生成后，再上 residual flow：

```text
x1_res = I_crack - I_pseudo
x0_res ~ N(0, I)
t ~ Uniform(0, 1)
x_t = (1 - t) * x0_res + t * x1_res
v_target = x1_res - x0_res
```

模型：

```text
ResidualFlowUNet(x_t, t, I_pseudo, M_gate, domain, style_condition) -> v_pred
```

loss：

```text
L_res_fm = MSE(v_pred, v_target)
```

采样：

```text
x_0 ~ N(0, I)
ODE integrate 0 -> 1
Delta_crack = x_1
I_syn = I_normal + gate(M_syn) * Delta_crack
```

### 9.5 V3：Tile-level Residual DiT

不要直接在 dphone 全图上做 full-attention DiT。dphone 为 1408x2560，patch size 16 会产生约 14080 tokens，代价太高。

更合理：

```text
tile-level 768 / 1024 DiT
latent residual DiT
mask-local crop DiT
```

DiT 只作为后续增强，不作为第一版主干。

---

## 10. Gate 设计

`gate(M)` 决定 residual 的生效范围。

### 10.1 三种 mask

```text
M_raw      # 原始 binary label，给 U-Net 用
M_band     # dilated mask，用于 residual loss inside band
M_gate     # dilated + blurred mask，用于 alpha blending
```

### 10.2 domain / resolution 自适应 dilation

不要固定 radius。建议：

```text
640x640:       radius 2-5
1920x1080:     radius 4-9
1408x2560:     radius 5-12
```

也可以按裂缝厚度自适应：

```text
r = clamp(2 * thickness_p90, min_r, max_r)
```

### 10.3 label 不变

最终保存给 downstream U-Net 的 mask 永远是：

```text
Y_syn = M_raw or decoded binary M_syn
```

不能保存 blur 后的 gate。

---

## 11. Synthetic Generation

### 11.1 输入

每次生成：

```text
normal image I_normal
same-domain mask M_syn
style noise z_style
area / thickness / orientation condition
renderer checkpoint
```

### 11.2 输出

```text
I_syn
M_syn
metadata
```

### 11.3 同域生成逻辑

```text
for domain in [camera, phone, dphone]:
    normals = normal images from domain
    mask_prior = domain-specific or domain-conditioned mask generator
    renderer = shared renderer + domain embedding / domain adapter
    generate synthetic samples
```

推荐合成数量第一版：

```text
camera: 300-600
phone:  300-600
dphone: 600-1200
```

不要一开始上万张。先证明 0.5x / 1x / 2x synthetic 有效，再扩量。

### 11.4 每个 normal 多候选

```text
for each normal:
    sample 2-4 masks
    for each mask:
        sample 2-4 residual styles
    keep candidates after filter
```

---

## 12. Synthetic Filter

生成样本必须筛。不要把所有 synthetic 都喂给 U-Net。

### 12.1 基础过滤

```text
mask_area_ratio within domain distribution
connected_components <= threshold
skeleton_length >= minimum
thickness <= domain threshold
bbox not outside valid area
```

### 12.2 图像一致性过滤

```text
outside_change = L1(I_syn * (1 - M_gate), I_normal * (1 - M_gate))
residual_leak  = mean(abs(Delta) * (1 - M_gate))
inside_contrast = contrast_gain(I_syn, I_normal, M_raw)
edge_gain = gradient_gain(I_syn, I_normal, M_raw)
```

丢弃：

```text
mask 外变化大
裂缝不可见
裂缝变成黑块 / blob
裂缝边缘太糊
背景出现奇怪纹理
```

### 12.3 Teacher consistency

用 Teacher 预测 synthetic：

```text
P_teacher = Teacher(I_syn)
```

计算：

```text
teacher_dice = Dice(P_teacher, M_syn)
teacher_recall_on_mask
teacher_fp_outside_mask
```

保留条件：

```text
teacher_recall_on_mask 高
teacher_fp_outside_mask 低
teacher_dice 达到 domain-specific threshold
```

注意：teacher 不应该作为唯一标准。teacher 看不懂的 hard case 不一定没用，但第一版可以先保守筛选。

### 12.4 总评分

```text
score =
  + w1 * teacher_dice
  + w2 * teacher_recall_on_mask
  + w3 * inside_contrast
  + w4 * edge_gain
  - w5 * outside_change
  - w6 * residual_leak
  - w7 * topology_penalty
  - w8 * area_distribution_penalty
```

建议按 domain 保留 top：

```text
camera: top 30%-50%
phone:  top 30%-50%
dphone: top 40%-60%
```

phone normal 太少，filter 要更保守。

### 12.5 输出

```text
runs/u_cdmr_flow_residual_plus/synthetic/raw/synthetic_manifest.csv
runs/u_cdmr_flow_residual_plus/synthetic/filtered/synthetic_filtered.csv
```

filtered manifest 字段：

```text
synthetic_image_path
synthetic_mask_path
source_normal_path
domain
mask_source
residual_source
seed_mask
seed_residual
style_seed
area_ratio
bbox_w
bbox_h
skeleton_length
component_count
thickness_mean
teacher_dice
teacher_recall_on_mask
teacher_fp_outside_mask
outside_change
residual_leak
inside_contrast
edge_gain
quality_score
renderer_checkpoint
mask_checkpoint
sampling_config
```

---

## 13. Downstream U-Net 训练

### 13.1 数据组成

只用：

```text
real crack + real mask
real normal + empty mask
filtered synthetic crack + synthetic mask
```

不使用 broken。

### 13.2 synthetic ratio 消融

不要一开始用 1:10 synthetic。建议：

```text
A. real only
B. real + copy-paste
C. real + mask bank + deterministic residual
D. real + descriptor mask flow + deterministic residual
E. real + mask bank + residual flow
F. real + latent mask flow + residual flow
```

每个方案跑：

```text
synthetic ratio = 0.5x
synthetic ratio = 1.0x
synthetic ratio = 2.0x
```

即：

```text
real crack : synthetic crack = 1 : 0.5
real crack : synthetic crack = 1 : 1
real crack : synthetic crack = 1 : 2
```

如果 2x 仍然稳定提升，再尝试 5x。

### 13.3 评估指标

必须报告：

```text
Dice
IoU
Precision
Recall
Boundary F1
small-crack recall
normal false-positive rate
per-domain Dice / IoU / Recall
per-area-bucket metrics
```

重点关注：

```text
small-crack recall 是否提升
normal false positive 是否上升
phone domain 是否过拟合
synthetic 是否只提升 train-like case
```

### 13.4 split

推荐默认：

```text
domain-stratified split
```

如果 `video_name` 可靠，优先：

```text
domain + video-level split
```

如果你已经人工去重、相邻帧风险清理过，可以做 image-level split，但论文里要说明。

---

## 14. 推荐训练阶段

### Phase 0：数据准备

```text
输入 manifest
过滤 broken
生成真实 mask
统计 domain distribution
生成 train / val / test split
```

输出：

```text
splits/train.csv
splits/val.csv
splits/test.csv
masks/*.png
stats/*.csv
```

### Phase 1：Teacher Segmenter

训练：

```text
real crack + real mask
real normal + empty mask
```

输出：

```text
teacher checkpoint
teacher validation report
normal false-positive maps
```

### Phase 2：Pseudo-normal

```text
real crack + mask -> pseudo-normal
```

输出：

```text
pseudo_manifest.csv
```

### Phase 3：Residual Renderer V1

```text
I_pseudo + M_real + domain + style_noise -> Delta_crack
I_recon = I_pseudo + gate(M_real) * Delta_crack
loss(I_recon, I_crack)
```

输出：

```text
residual_renderer_v1 checkpoint
reconstruction samples
```

### Phase 4：Mask Generator V1

```text
mask bank
optional descriptor flow
```

输出：

```text
sampled mask manifest
```

### Phase 5：Synthetic Generation + Filter

```text
same-domain normal + sampled mask + renderer -> synthetic pair
filter by quality
```

输出：

```text
synthetic_raw.csv
synthetic_filtered.csv
quality_report.md
```

### Phase 6：Downstream U-Net

```text
real-only
real + synthetic 0.5x
real + synthetic 1x
real + synthetic 2x
```

输出：

```text
downstream checkpoints
metrics table
per-domain report
```

### Phase 7：Flow 增强

在 Phase 1-6 跑通后，再做：

```text
latent dense mask flow
residual flow matching
tile-level residual DiT
```

---

## 15. 推荐 repo 结构

```text
src/ucdmr_flow_residual_plus/
  data/
    manifest_dataset.py
    labelme_to_mask.py
    mask_representations.py
    domain_stats.py
    split.py
    pseudo_normal.py

  models/
    encoders.py
    teacher_segmenter.py
    mask_descriptor_flow.py
    mask_autoencoder.py
    mask_latent_flow.py
    residual_renderer.py
    residual_flow.py
    domain_embedding.py
    gates.py

  losses/
    segmentation_losses.py
    residual_losses.py
    mask_losses.py
    topology_losses.py

  train/
    train_teacher.py
    train_residual_renderer.py
    train_mask_descriptor_flow.py
    train_mask_autoencoder.py
    train_mask_latent_flow.py
    train_residual_flow.py
    train_downstream.py

  sample/
    sample_masks.py
    generate_synthetic_pairs.py
    filter_synthetic_pairs.py

  eval/
    eval_teacher.py
    eval_downstream.py
    eval_synthetic_quality.py

  utils/
    image_io.py
    geometry.py
    visualization.py
    metrics.py
    seed.py
```

配置文件：

```text
configs/ucdmr_plus/data.yaml
configs/ucdmr_plus/teacher.yaml
configs/ucdmr_plus/pseudo_normal.yaml
configs/ucdmr_plus/residual_renderer.yaml
configs/ucdmr_plus/mask_descriptor_flow.yaml
configs/ucdmr_plus/mask_latent_flow.yaml
configs/ucdmr_plus/residual_flow.yaml
configs/ucdmr_plus/generate.yaml
configs/ucdmr_plus/filter.yaml
configs/ucdmr_plus/downstream.yaml
```

---

## 16. 推荐命令顺序

### 16.1 最小可跑通版本

```bash
python -m src.ucdmr_flow_residual_plus.data.split \
  --manifest /path/to/manifest_merged.csv \
  --ignore-label broken \
  --out runs/u_cdmr_flow_residual_plus/splits

python -m src.ucdmr_flow_residual_plus.data.labelme_to_mask \
  --manifest runs/u_cdmr_flow_residual_plus/splits/train.csv \
  --out runs/u_cdmr_flow_residual_plus/masks

python -m src.ucdmr_flow_residual_plus.train.train_teacher \
  --config configs/ucdmr_plus/teacher.yaml

python -m src.ucdmr_flow_residual_plus.data.pseudo_normal \
  --config configs/ucdmr_plus/pseudo_normal.yaml

python -m src.ucdmr_flow_residual_plus.train.train_residual_renderer \
  --config configs/ucdmr_plus/residual_renderer.yaml

python -m src.ucdmr_flow_residual_plus.sample.generate_synthetic_pairs \
  --config configs/ucdmr_plus/generate.yaml \
  --mask-source bank \
  --residual-source renderer_v1

python -m src.ucdmr_flow_residual_plus.sample.filter_synthetic_pairs \
  --config configs/ucdmr_plus/filter.yaml

python -m src.ucdmr_flow_residual_plus.train.train_downstream \
  --config configs/ucdmr_plus/downstream.yaml \
  --use-synthetic false

python -m src.ucdmr_flow_residual_plus.train.train_downstream \
  --config configs/ucdmr_plus/downstream.yaml \
  --use-synthetic true \
  --synthetic-ratio 1.0

python -m src.ucdmr_flow_residual_plus.eval.eval_downstream \
  --config configs/ucdmr_plus/downstream.yaml
```

### 16.2 加 descriptor mask flow

```bash
python -m src.ucdmr_flow_residual_plus.train.train_mask_descriptor_flow \
  --config configs/ucdmr_plus/mask_descriptor_flow.yaml

python -m src.ucdmr_flow_residual_plus.sample.generate_synthetic_pairs \
  --config configs/ucdmr_plus/generate.yaml \
  --mask-source descriptor_flow \
  --residual-source renderer_v1
```

### 16.3 加 residual flow

```bash
python -m src.ucdmr_flow_residual_plus.train.train_residual_flow \
  --config configs/ucdmr_plus/residual_flow.yaml

python -m src.ucdmr_flow_residual_plus.sample.generate_synthetic_pairs \
  --config configs/ucdmr_plus/generate.yaml \
  --mask-source descriptor_flow \
  --residual-source residual_flow
```

---

## 17. 关键 ablation 表

建议论文 / 实验报告里至少做：

| ID | 训练数据 | Mask 来源 | Residual 来源 | 目的 |
|---|---|---|---|---|
| A | real only | - | - | 下游基线 |
| B | real + copy-paste | real mask | alpha blend | 传统增强基线 |
| C | real + synthetic | mask bank | deterministic renderer | 检查 residual 主线是否有效 |
| D | real + synthetic | descriptor flow | deterministic renderer | 检查 mask flow 是否有效 |
| E | real + synthetic | mask bank | residual flow | 检查 residual flow 是否有效 |
| F | real + synthetic | descriptor flow | residual flow | 完整 V1 flow 方案 |
| G | real + synthetic | latent mask flow | residual flow | 完整增强版 |

每个都报告：

```text
overall Dice / IoU / Recall
camera Dice / Recall / FP
phone Dice / Recall / FP
dphone Dice / Recall / FP
small-crack recall
normal false-positive rate
```

---

## 18. 哪些东西第一版不要做

不要第一版就做：

```text
full RGB generation
full-image DiT on dphone
teacher-guided ODE gradient sampling
10 个 loss 同时上
1:10 synthetic ratio
cross-domain image style generation
broken hard negative
```

原因：

```text
数据量不够
调参困难
容易生成看似合理但对 U-Net 有害的数据
无法判断哪个模块真的贡献了性能
```

第一版只做：

```text
pseudo-normal
mask bank / descriptor flow
pretrained encoder residual renderer
same-domain generation
synthetic filter
downstream closed-loop evaluation
```

---

## 19. 最终推荐版本

最终主方法建议定为：

```text
U-CDMR-Flow-Residual+
= Same-domain mask-first residual synthesis
+ pretrained visual encoder
+ descriptor / latent mask flow
+ stochastic residual renderer / residual flow
+ segmentation-aware synthetic filtering
+ downstream U-Net closed-loop evaluation
```

方法主张：

```text
裂缝合成不应该生成整张 RGB 图，
而应该生成和 mask 严格耦合的局部 residual。
```

最终输出：

```text
synthetic image
synthetic binary mask
metadata
quality score
```

最终目标：

```text
提升真实测试集上的裂缝分割能力，尤其是 tiny crack recall，
同时控制 normal false positive。
```

---

## 20. 最短实施路线

如果现在就开始做，按这个顺序：

```text
1. 过滤 broken，重建 crack/normal manifest
2. LabelMe -> mask + skeleton + SDF + gate
3. 训练 Teacher Segmenter
4. 生成 pseudo-normal
5. 训练 pretrained-encoder residual renderer
6. 用真实 mask bank + renderer 生成第一批 synthetic
7. filter synthetic
8. 训练 downstream real-only vs real+synthetic
9. 如果有效，再加 descriptor mask flow
10. 如果仍有效，再加 residual flow
```

不要等所有 fancy 模块都实现完才开始下游实验。第一批结果必须尽快回答：

```text
合成数据到底能不能提高真实 U-Net 分割？
```

这个问题没回答前，任何更复杂的 flow / DiT 都是次要的。
