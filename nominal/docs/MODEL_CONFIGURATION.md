# 运行时模型配置与扩展指南

本文件说明本次重构后如何增删 Resonance/Term、如何注册新的 GVV Wave，
以及未来把工程改造成 GPPP 等其他衰变道时应保留或替换哪些部分。本次重构
没有加入新的共振态，也没有实现 `2++` Wave。

## 1. 唯一的模型描述

`config/model.json` 是用户维护的唯一振幅模型描述。程序不再在 C++ 中保存一份
名义共振态数、Term 数或拟合参数顺序。启动时执行以下流程：

```text
model.json
  -> framework/ModelDefinition：语法、字段、稳定 id、参考振幅检查
  -> process/GVVProcessModel：GVV propagator/Wave/dynamics 检查与稠密索引编译
  -> GVVFitParameters：从编译模型生成唯一的 Minuit 参数布局
  -> GVVSample/kernel：按运行时 Wave/Term 数分配 GPU 缓冲并计算强度
  -> NLL_estimator：MC 归一化及带符号样本似然
  -> Fit/PostFit/projection：按同一模型元数据读写结果
```

JSON Schema 位于 `config/model.schema.json`。C++ loader 也会拒绝未知字段，因而
例如把 `width` 拼成 `widht` 会在拟合开始前失败，而不会被静默忽略。持久化的
物理身份一律使用字符串 `id`；整数索引只是在一次运行中生成的 GPU 稠密布局。

## 2. Resonance、Wave 与 Term

- **Resonance**：一组线形/传播子参数，例如质量、宽度、轨道角动量或 Flatte
  比值。它本身不决定协变张量。
- **Wave**：完整的 GVV 协变张量基底及其 `J^PC` 元数据。Wave 是 CUDA 代码，
  因此新增 Wave 仍需显式注册和测试。
- **Term**：把一个 Resonance、一个 Wave、一个复耦合以及本衰变道的 dynamics
  组合成进入相干求和的振幅项。同一个 Resonance 可以对应多个 Wave/Term。

事件强度的运行时合同为

```text
d_i(event) = coupling_i * GVV_dynamics_i(event)
I(event) = sum_ij Re[d_i d_j* F_(wave_slot_i,wave_slot_j)(event)]
```

`F` 只对模型实际使用的 Wave 建立紧凑矩阵；`d_i` 和分量矩阵按实际 Term 数
建立。因此增删 Term 不需要修改数组维度、枚举上限、Minuit 参数数或 PostFit
循环上限。

## 3. 增加一个使用现有 Wave/propagator 的共振态

只编辑 `config/model.json`：

1. 在 `resonances` 中加入具有唯一 `id` 的对象；
2. 在 `terms` 中加入一个或多个对象，并在 `dynamics.resonance` 中引用该 id；
3. 选择已注册的 `wave`，设置 `coupling`；
4. 构建并至少运行 `test_model_definition.exe` 与
   `test_gvv_process_model.exe`，再开始正式拟合。

示意配置如下，数值仅用于说明格式：

```json
{
  "id": "example_x",
  "label": "example X",
  "propagator": "two_body_running_bw",
  "parameters": {
    "mass": {"value": 2.0, "fixed": true},
    "width": {"value": 0.1, "fixed": true},
    "orbital_l": {"value": 1, "fixed": true}
  }
}
```

```json
{
  "id": "example_x_11",
  "label": "X_{example}",
  "wave": "gvv.pseudoscalar_11",
  "coupling": {
    "mode": "complex_cartesian",
    "initial": [0.1, 0.0]
  },
  "dynamics": {
    "type": "gvv_x_to_omega_omega",
    "resonance": "example_x"
  }
}
```

目前 GVV process compiler 支持以下传播子名称：

| `propagator` | 必需参数 | 说明 |
|---|---|---|
| `nonresonant` | 无 | 常数 X 因子 |
| `fixed_width_bw` | `mass`, `width` | 固定宽度 BW |
| `two_body_running_bw` | `mass`, `width`, `orbital_l` | 当前支持 `L=0,1` |
| `scalar_sd_running_bw` | `mass`, `width`, `sd_ratio` | `sd_ratio` 使用 log 变换 |
| `subtracted_effective_flatte` | `mass`, `width`, `omegaomega_ratio` | 比值使用 log 变换 |

传播子数学函数位于 `include/Dynamics.h`；GVV 对 JSON 参数的注册与检查集中在
`src/GVVProcessModel.cu`。新增传播子时应在这两处分别加入纯数学实现和显式注册，
并在 `tests/test_dynamics.cu`、`tests/test_gvv_model.cu` 中增加数值/边界测试。

## 4. 删除或暂时关闭 Term

探索模型时可先给 Term 添加：

```json
"active": false
```

inactive Term 不会进入 GPU 布局、拟合参数布局、projection 或 PostFit。模型确定后
建议删除不再使用的 Term；若某 Resonance 不再被任何 active Term 引用，也应一并
删除，避免保留与似然无关的可浮动参数。

参考振幅约束必须始终满足：

- 全部 active Term 中恰好一个 `scale_and_phase` 参考，它使用
  `fixed_complex` 和 `[1.0, 0.0]`；
- 每个实际相干类恰好一个相位参考；`positive_real` 对应 `phase`，其模长以
  `log(rho)` 拟合；
- 普通自由复耦合使用 `complex_cartesian`，不能声明 reference。

因此，关闭参考 Term 前必须把同一相干类中的另一个 Term 改成相应参考。loader
会在启动阶段报告缺失或重复参考，而不是让 Minuit 拟合一个不可辨识模型。

## 5. 注册新的 GVV Wave

新增 Wave 不是纯 JSON 操作，因为其协变张量需要由 nvcc 编译。本次没有实现
`2++`；未来加入它时按以下唯一入口操作：

1. 在 `include/GVVAmplitude.h` 实现完整 tensor，并在 `gvv_wave_tensor` 的 device
   dispatch 中分配稳定的 `GVVWaveType`；
2. 在 `src/GVVProcessModel.cu` 的 `gvv_wave_registry()` 加入字符串 id、`J^PC`、
   LaTeX 标签、coherence class 与 device wave type；
3. 在 `model.json` 的 Term 中引用新的字符串 Wave id；
4. 在 `tests/test_gvv_amplitude.cu` 加编译/张量数值测试，在
   `tests/test_gvv_process_model.cu` 加 registry/编译测试；
5. 检查该 `J^PC` 相干类的参考振幅策略，再运行全量测试。

完整 GVV Wave 依赖两个 omega 衰变流和光子投影，因此属于 `process`，不能冒充
跨衰变道通用 Wave。可以跨项目复用的是 `FV.h`、`Tensor.h`、`Dynamics.h` 中的
Lorentz/轨道张量与传播子数学积木。

## 6. 耦合与拟合参数

Term 的 `coupling.mode` 决定参数生成规则：

| mode | 生成的拟合参数 |
|---|---|
| `complex_cartesian` | `Re_<term-id>`, `Im_<term-id>` |
| `positive_real` | `log_rho_<term-id>` |
| `fixed_complex` | 无 |

Resonance 参数只有在 JSON 中 `fixed: false` 且被相应 propagator 明确支持时才加入
参数布局。参数名称、顺序、初值、步长、边界、结果解析和 covariance 维度都来自
`gvv_fit_parameter_layout()`，不得在 Fit 或 PostFit 中建立第二份顺序表。

提交默认模型：

```bash
./scripts/Sub.sh 10 20260815 config/model.json
```

保留多个候选模型时可复制 JSON 并显式传入：

```bash
./scripts/Sub.sh 20 20260815 config/models/model_without_x.json
```

Fit 会把实际使用的 canonical JSON 写成
`<fit_result>.model.json`。PostFit 默认优先读取这份快照，从而避免后来修改
`config/model.json` 后误用不同模型。直接调用程序时完整接口为：

```text
Fit.exe data.root normalization_mc.root SB1.root SB2.root \
  [fit_result.txt [n_starts [base_seed [model.json]]]]

PostFit.exe fit_result.txt Cova_matrix.dat truth_mc.root \
  normalization_mc.root [output_prefix [model.json]]
```

## 7. 似然、projection 与 PostFit

`include/framework/Likelihood.h` 只接收事件强度，负责 MC 均值归一化、PDF 有效性
检查和带符号 log-likelihood contribution；它不包含 GVV、ROOT branch 或 sideband
名称。`NLL_estimator` 是当前 GVV 应用的编排器，负责把 process intensity 提供给
这个通用层。sideband 系数属于 Fit 配置，不属于 Resonance/Wave/Term 模型。

projection 的 `weight_component` 是运行时 `n_terms * n_terms` 向量，ROOT 文件同时
写出 `component_map`、`group_map` 和 `n_terms/n_groups`。PostFit 的分量对、`J^PC`
分组、可观测量和 covariance 维度也从模型元数据生成。旧的 `weight_0pp`、
`weight_0mp` 分支仅作为当前名义画图宏的兼容别名，不再控制模型结构。

## 8. 将工程改成新的衰变道

这里的“支持 GPPP”不是在同一可执行文件中动态装载第二个 channel，而是创建一个
新的、单衰变道项目。迁移时的边界如下：

保留并复用：

- `framework/ModelDefinition` 与 JSON schema 思路；
- `framework/Likelihood`；
- `FV.h`、`Tensor.h`、`Dynamics.h` 中适用的数学积木；
- 运行时稠密索引、参数描述、结果模型快照、component closure 的设计模式。

替换或重构：

- GVV ROOT event/branch schema 与 kinematics cache；
- `GVVAmplitude.h` 和 GVV Wave registry；
- `GVVProcessModel` 中的 process id、Term dynamics parser/device representation；
- omega/rho 专属传播链、Term coefficient kernel；
- GVV projection 变量、标签和画图宏；
- GVV process tests。

新项目的 framework 不应 include 任何 GVV/GPPP header；process 可以依赖 framework
和数学积木。这个单向依赖是判断边界是否正确的最简单标准。

## 9. 修改后的最小验证

```bash
make -j4
make tests -j4
./tests/bin/test_model_definition.exe config/model.json
./tests/bin/test_gvv_process_model.exe config/model.json
./tests/bin/test_gvv_fit_parameters.exe
./tests/bin/test_likelihood.exe
./bin/PostFit.exe --self-test
bash -n scripts/*.sh config/gvv_env.sh
```

`test_gvv_process_model` 同时编译 6-Term 与 8-Resonance/8-Term 的内存模型，专门
防止名义的 7/7 数量重新以隐藏常量形式进入生产代码。
