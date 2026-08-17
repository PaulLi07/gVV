# 配置说明

项目只有两个用户配置入口：`model.json` 描述物理模型，`fit.json` 描述一次拟合运行。没有生成 JSON 的 Python 胶水层。

## model.json

顶层字段：

- `schema_version`：当前为 `1`；
- `process`：必须为当前过程编译器接受的 `psi2s_to_gamma_omega_omega`；
- `metadata`：模型名字和说明；
- `resonances`：传播子实例；
- `terms`：参与相干和的振幅项。

### Resonance

```json
{
  "id": "f0_1710",
  "label": "f0(1710)",
  "propagator": "two_body_running_bw",
  "parameters": {
    "mass": {"value": 1.723, "fixed": true},
    "width": {"value": 0.149, "fixed": true},
    "orbital_l": {"value": 0, "fixed": true}
  }
}
```

当前传播子配置契约如下；缺少参数或额外/拼错的参数都会在 GPU 分配之前报错：

| `propagator` | 允许的 `parameters` |
|---|---|
| `nonresonant` | 空对象 |
| `fixed_width_bw` | 固定 identity `mass`、`width` |
| `two_body_running_bw` | 固定 identity `mass`、`width`、整数 `orbital_l`（当前 0 或 1） |
| `scalar_sd_running_bw` | 固定 identity `mass`、`width`；正且 log 变换的 `sd_ratio` |
| `subtracted_effective_flatte` | 固定 identity `mass`、`width`；正且 log 变换的 `omegaomega_ratio` |

通用公式位于 `framework/dynamics/`；上表的字符串、允许字段和 GVV 参数策略只在 `process/WaveRegistry.cu` 映射。

参数对象支持 `value`、`fixed`、`transform`、`step` 和 `bounds`。当前过程允许拟合的正参数使用 `transform: "log"`，因此 Minuit 空间不会进入非物理负值。对于 log 变换参数，`value` 是物理空间初值，而 `step` 和 `bounds` 是 `log(value)` 的 Minuit 空间量。

### Term

```json
{
  "id": "f0_1710_00",
  "label": "f_{0}(1710)",
  "wave": "gvv.scalar_00",
  "coupling": {
    "mode": "positive_real",
    "reference": "phase",
    "initial": 0.1
  },
  "dynamics": {
    "type": "gvv_x_to_omega_omega",
    "resonance": "f0_1710"
  }
}
```

`active: false` 可临时关闭一个 Term。耦合模式：

- `complex_cartesian`：自由 `Re/Im`；
- `positive_real`：相位参考，拟合 `log(rho)`；
- `fixed_complex`：尺度和相位参考，通常固定为 `1+0i`。

每个 `coherence_class` 必须恰有一个相位参考。该类来自 Wave 注册表，不由用户在每个 Term 重复填写。

当前 GVV Term 的 `dynamics` 必须且只能包含字符串字段 `type` 与 `resonance`；`type` 必须为 `gvv_x_to_omega_omega`，`resonance` 必须引用已定义的 Resonance id。这样拼写错误不会变成静默无效配置。

### 增删共振态

增加一个已有 Wave 上的共振项时，同时增加 Resonance 定义与引用它的 Term。删除时同时删除相应 Term；没有 Term 引用的 Resonance 不进入相干和，但建议一起清理。无需修改 C++ 总数、参数总数或 CUDA 数组长度。

## fit.json

```json
{
  "schema_version": 1,
  "model": "config/model.json",
  "inputs": {
    "data": "RootSet/data.root",
    "normalization_mc": "RootSet/normalization_mc.root",
    "backgrounds": [
      {"label": "SB1", "file": "RootSet/SB1.root", "coefficient": -0.5},
      {"label": "SB2", "file": "RootSet/SB2.root", "coefficient": 0.25}
    ]
  },
  "minimizer": {
    "n_starts": 10,
    "base_seed": 20260815,
    "maximum_edm": 0.001,
    "maximum_calls": 20000,
    "tolerance": 0.1,
    "error_definition": 0.5,
    "random_magnitude": [0.05, 5.0]
  },
  "output": {
    "directory": "results",
    "log_directory": "runlog",
    "tag": "initial"
  }
}
```

带权样本按 `lnL_eff += coefficient * sum(log(P))` 进入似然。当前二维边带的 `-0.5/+0.25` 因而完全在配置层表达。

`n_starts` 的第 0 个起点使用模型初值；后续起点只随机化自由复耦合的模和相位，传播子物理参数保持名义起点。`output.tag` 只允许字母、数字、点、下划线和连字符；相同 tag 的四个输出会覆盖。
