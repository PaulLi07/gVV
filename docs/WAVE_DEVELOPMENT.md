# 新 Wave 开发指南

本次重构只迁移现有 Wave，没有实现新的 `2++`。以下接口是后续添加新 Wave 时的稳定边界。

## 1. 先确认完整物理定义

一个 Wave 文件应表达当前过程的完整协变张量基底，包括辐射产生、`X -> omega omega` 角动量耦合、需要的投影算符和障碍因子。传播子不属于 Wave；同一 Wave 可被多个 Resonance/Term 复用。

## 2. 复用积木

优先从这些目录组合：

- `framework/math/`：四矢量、度规、Levi-Civita、设备复数；
- `framework/tensors/`：SpinProjector、OrbitalTensor、BarrierFactor、Tensor；
- `framework/dynamics/`：二体动量和传播子；
- `process/ProcessKinematics.cuh`：本过程的组合四动量和 omega 常量。

若缺少真正通用的张量积木，应把它加到 `framework/tensors/` 并写独立测试；若公式只对 GVV 成立，应留在 `process/`。

## 3. 新建单一 Wave 文件

在 `process/waves/` 新建有物理含义的 `.cuh`，对外暴露一个纯设备函数：

```cpp
__device__ inline tensor gvv_example_tensor(
    const GVVEventKinematics& event,
    const GVVBarrierParameters& barrier = GVVBarrierParameters());
```

函数不得访问全局可变状态、Term 编号或 Resonance 个数。

## 4. 在唯一位置注册

在 `WaveRegistry.cuh`：

1. 给 `GVVWaveType` 增加稳定枚举；
2. 在 `gvv_wave_tensor` 增加设备 dispatch；
3. 更新 `GVV_NBASIS`。

在 `WaveRegistry.cu` 的 `gvv_wave_registry()` 增加字符串 id、JPC、LaTeX、相干类和枚举值。这是唯一的主机注册表；禁止在 Fit、TermEvaluator 或 model parser 再写第二份映射。

注册完成后，`ProcessAmplitude.cuh` 会通过统一 dispatch 取得新 Wave，并沿用
既有光子偏振投影、Wave 缩并和 `F_ij` 组装；通常不需要修改该文件。只有过程的
偏振求和或完整振幅缩并规则本身发生变化时，才调整 `ProcessAmplitude`。

## 5. 测试后再配置模型

至少增加：

- Wave 设备编译/有限值测试；
- 注册 id 到枚举的测试；
- 新 Wave 与自身及已有 Wave 的 F 矩阵收缩测试；
- 一个最小 `model.json` 编译测试。

完成注册以后，用户才能在 `model.json` 的 Term 中引用新 id。Resonance 的增删仍然只发生在 JSON。

## 6. 审核边界

提交前检查：

- Wave 文件没有传播子参数和具体 Resonance 名称；
- framework 没有新增 GVV include；
- 新枚举只在 WaveRegistry 注册一次；
- 设备 dispatch 对未知枚举明确失败或返回受控零值；
- `make check` 和至少一次 GPU 端等价/有限值核验通过。
