# 架构与数据流

## 设计目标

本架构解决三个不同频率的变化：

- 高频：增加、删除、启停共振态，只编辑 `model.json`；
- 中频：增加过程允许的新 Wave，只修改 `process/waves/` 和 Wave 注册表；
- 低频：把项目改造成另一个末态时，保留 `framework/`，重写过程层。

它不试图构建一个同时支持所有末态的庞大运行时框架。过程物理保持显式，通用算法保持可复用。

## 模块依赖方向

```text
config/model.json ──> framework/model ──> process/WaveRegistry
                                             │
ROOT samples ──> process/SampleLoader ──> process/TermEvaluator
                                             │
                    framework/likelihood <── process/FitLikelihood
                                             │
config/fit.json ──> framework/fit ───────> app/Fit.cu
                                             │
                                 txt + covariance + ROOT projection
```

硬约束是 `framework/` 不包含 `process/` 头文件。过程层可以依赖框架层，反方向不允许。
框架内部的物理工具也保持单向依赖：`math <- tensors <- dynamics <- process`。
`BarrierFactor` 自己拥有单位与默认半径，`dynamics` 可以调用障碍因子，但
`tensors` 不再反向包含 `dynamics`。

## framework

### math 与 tensors

`math/` 提供四矢量、复数、度规和 Levi-Civita 约定；`tensors/` 提供自旋投影、轨道张量、障碍因子和张量收缩积木。它们不包含 omega、GVV Wave 编号或 ROOT 分支名。

### dynamics

`Propagators.cuh` 提供传播子公式，`Kinematics.cuh` 提供二体运动学，`PropagatorRegistry.cuh` 定义设备侧传播子类型与统一求值入口。GVV 仅在过程编译器中把 JSON 名称映射到这些通用类型。

### model

`Model.cpp` 严格解析唯一的模型描述层：Resonance、Term、耦合和传播子参数。未知字段、重复 id、非法引用和非有限数在 GPU 分配之前失败。这里不注册 GVV Wave。

### amplitude 与 likelihood

`IntensityEngine.cuh` 只负责相干和：

```text
A_i(x) = c_i D_i(s) U_i(x)
I(x)   = sum_ij c_i c_j* D_i D_j* F_ij(x)
```

`Likelihood.h` 实现通用 MC 归一化与带权无分箱对数似然算术。GVV 二维边带系数只是 `fit.json` 中的带权样本，不硬编码进框架。

### fit

- `FitConfig`：严格读取一次拟合的输入、最小化和输出标签；
- `FitEngine`：通用多起点 TMinuit 驱动，只接受参数规格和目标函数回调；
- `FitOutput`：写统一参数行和协方差，不理解 Resonance 或 Wave。

`FitEngine` 的参数数目来自传入 vector，绝无 `NRES`、`NTERM` 或固定的总参数数。

## process

### 末态与输入边界

`ProcessEvent.cuh` 定义七个末态粒子的设备表示；`ProcessKinematics.cuh` 定义 omega/psi 过程常量与组合；`SampleLoader` 是 ROOT `Pwa` 树到过程事件的唯一映射点。

### Wave 边界

每个 `process/waves/*.cuh` 表示一个完整的过程 Wave，而不是一段传播子或一个 Resonance：

- `Scalar00.cuh`：现有 `0++(00)`；
- `Scalar22.cuh`：现有 `0++(22)`；
- `Pseudoscalar11.cuh`：现有 `0-+(11)`。

`WaveRegistry.cuh` 是设备 dispatch 的唯一位置；`WaveRegistry.cu` 是字符串 id、JPC、相干类与数值 wave type 的唯一主机注册位置。

当前 `WaveRegistry.cuh` 还包含三段过程振幅公共计算：光子偏振投影
`gvv_photon_projector`、两个 Wave 的偏振缩并 `gvv_wave_contraction`，以及组装
`F_ij` 的 `gvv_cal_F`。它们不是注册元数据；若后续继续收窄注册表，应整体移到
`process/ProcessAmplitude.cuh`，由 `TermEvaluator` 包含。枚举、设备 dispatch、
Wave 元数据和编译模型类型仍留在 `WaveRegistry`。本轮只记录这个边界，不扩大改动范围。

### Resonance、Wave、Term

- Resonance：传播子实例和动力学参数；
- Wave：与末态和角动量耦合有关的完整张量基底；
- Term：一个 Resonance、一个 Wave 和一个复耦合的连接。

`gvv_compile_model` 把稳定字符串 id 编译为紧凑运行时索引。禁用 Term 不占设备数组；增加 Term 会自动扩大 GPU 缓冲、Minuit 参数布局与投影矩阵。

### 目标函数与参数映射

`FitLikelihood` 管理 GVV 样本、缓存的 F 矩阵、omega 宽度表和 GPU 模型状态，并调用通用似然算术。它不决定 Minuit 起点、收敛选择或输出命名。

它目前具体负责四组工作：加载并持有 data/normalization MC/带权背景；在
`Prepare()` 中构建 omega 宽度表、上传紧凑模型并缓存每个样本的 F 矩阵；每次
目标函数调用时同步可变传播子/耦合、计算 MC 归一化和有符号 log-likelihood；在
拟合完成后写包含运动学、分量权重和元数据的 projection ROOT。最后一项占据文件
的大部分辅助代码，是拟合到下游处理的输出桥梁，但不参与 Minuit 决策。

`ParameterMapping` 是唯一的模型状态到拟合参数转换层。它根据耦合参考约定和未固定传播子参数生成通用 `FitParameterSpec`，同时保存如何把 Minuit vector 写回 GVV 状态的 binding。

传播子设备描述 `ctpwa::PropagatorParameters` 只保存公式求值需要的数值。`sd_ratio` 或 Flatte ratio 是否参与拟合属于 GVV 模型编译策略，保存在过程层 metadata，不进入 `framework/dynamics/`。

## 代码阅读索引

建议按“配置 → 编译模型 → 样本 → 振幅 → 似然 → 拟合”的顺序阅读：

| 文件 | 职责 |
|---|---|
| `app/Fit.cu` | 可执行程序胶水；组装配置、过程似然、通用拟合与输出 |
| `framework/model/Model.*` | 严格解析通用 Resonance/Term/耦合描述 |
| `process/WaveRegistry.*` | 注册完整 GVV Wave，并把通用模型编译为设备紧凑布局 |
| `process/waves/*.cuh` | 每个文件实现一个完整过程 Wave |
| `process/SampleLoader.*` | ROOT 七末态分支到主机/GPU 数组的唯一入口 |
| `process/ProcessEvent.cuh` | 单个 GVV 事件在设备上的组合运动学视图 |
| `process/ProcessKinematics.cuh` | omega 三体衰变流和 GVV 过程常量 |
| `framework/dynamics/*` | 可复用二体运动学与传播子函数库 |
| `framework/tensors/*` | 可复用投影、轨道张量、障碍因子和低阶张量代数 |
| `process/TermEvaluator.*` | CUDA 上构造 Term 系数、F 矩阵和相干强度 |
| `process/FitLikelihood.*` | 样本编排、MC 归一化、有符号似然与 projection ROOT |
| `process/ParameterMapping.*` | `model.json` 状态与 Minuit vector 的唯一双向映射及物理参数 TXT 明细 |
| `framework/fit/FitEngine.*` | 与过程无关的多起点 MIGRAD/HESSE 驱动 |
| `framework/fit/FitOutput.*` | 统一 TXT 和 covariance 输出 |
| `submit.sh` | 唯一 Slurm 提交/worker 入口 |

源码顶部说明文件边界；关键公式、索引布局、单位、参考振幅和状态转换在实现位置附近说明。JSON 不允许注释，因此所有可配置字段统一记录在 `MODEL_CONFIGURATION.md`。

## 一次振幅拟合的完整流程

1. `FitConfig` 读取 `fit.json`，得到 `model.json`、样本、边带系数、Minuit 选项和输出 tag。
2. `Model` 严格解析 Resonance/Term；`WaveRegistry` 验证 Wave 并编译紧凑索引。
3. `SampleLoader` 读取 data、normalization MC 和带权背景样本。
4. 每个样本只构建一次与拟合参数无关的 Wave 收缩矩阵 `F_ij(x)`。
5. `ParameterMapping` 从编译模型生成 Minuit 参数 vector；参考振幅不进入自由参数。
6. 每次目标函数调用把 vector 写回复耦合或传播子参数，计算 normalization MC 的积分，再计算 data 与边带的有效 log-likelihood。
7. `FitEngine` 对 nominal 起点和随机耦合起点依次执行 MIGRAD/HESSE，只从满足状态、协方差和 EDM 条件的结果中选择最小 NLL。
8. 最优状态写入一个详细 TXT、一个协方差矩阵和一个 ROOT projection；Slurm 标准输出写入同 tag 日志。

## 未来末态重构边界

构建例如光子加若干赝标量或其他末态的新项目时，应复用整个 `framework/`。需要替换的是：

- `ProcessEvent`、`ProcessKinematics` 与 `SampleLoader`；
- `process/waves/` 及其注册表；
- Resonance/Wave 到具体 Term 的过程编译与求值；
- 过程专属 projection 内容；
- `app/Fit.cu` 中很薄的输入契约胶水。

不应复制或改写传播子公式、通用张量、似然算术、多起点 Minuit 或统一输出协议。
