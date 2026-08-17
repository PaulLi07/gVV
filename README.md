# gVV

`gVV` 是 `psi(2S) -> gamma omega omega` 协变张量分波拟合项目。当前版本把可复用的数理与拟合设施放在 `framework/`，把本衰变过程的末态、完整 Wave、Term 计算和 ROOT 输入约定放在 `process/`。在已有 Wave 范围内，改变模型只需要编辑 `config/model.json`，不再修改或重新计数 C++ 数组。

## 日常使用

```bash
source config/gvv_env.sh
make -j2
make check
./submit.sh config/fit.json
```

`Fit.exe` 也可以直接接受唯一的配置入口：

```bash
bin/Fit.exe config/fit.json
```

实际 GPU 拟合应通过 Slurm 提交，不要在登录节点运行。`submit.sh` 同时承担提交端和计算节点 worker 的职责，因此仓库中不再维护一组相互依赖的脚本。

输出名由 `config/fit.json` 的 `output.tag` 统一控制。同名文件直接覆盖：

- `results/fit_result-<tag>.txt`
- `results/Cova_matrix-<tag>.dat`
- `results/projection-<tag>.root`
- `runlog/fit-<tag>.log`

不会额外生成模型快照、拟合配置快照或 run 子目录。

## 模型修改边界

增加或删除已有 Wave 上的共振态时，只改 `config/model.json`：

1. 在 `resonances` 中定义传播子和参数；
2. 在 `terms` 中把该 Resonance、已注册 Wave 和复耦合连接起来；
3. 保证每个相干类恰有一个相位参考。

数组长度、Minuit 参数数、参数名称、GPU Term 布局和投影分量映射均由配置运行时生成。

增加新 Wave 时，用户只触及过程层：在 `process/waves/` 用 `framework/math/` 与 `framework/tensors/` 的积木实现完整过程 Wave，然后在 `process/WaveRegistry.cuh/.cu` 的唯一注册点登记。此次等价重构没有增加新的 `2++` Wave；`Scalar22.cuh` 是原有 `0++(22)` 基底。

未来改造为另一个末态时，目标不是让一个程序同时容纳所有衰变道，而是复用 `framework/`，替换 `process/` 和少量 `app/` 胶水。传播子、张量积木、通用似然算术、多起点 Minuit 驱动和输出协议不依赖 GVV。

## 目录

```text
gVV/
├── app/                    # Fit 可执行程序的薄胶水层
├── config/                 # model.json 与 fit.json
├── framework/              # 可迁移到其他末态的通用设施
│   ├── amplitude/
│   ├── dynamics/
│   ├── fit/
│   ├── likelihood/
│   ├── math/
│   ├── model/
│   └── tensors/
├── process/                # psi(2S)->gamma omega omega 专属实现
│   └── waves/
├── postfit/                # 原下游绘图样式，暂不在本次重构范围
├── tests/
├── docs/
├── Makefile
└── submit.sh
```

详细边界和调用流见 `docs/ARCHITECTURE.md`；模型字段见 `docs/MODEL_CONFIGURATION.md`；新 Wave 的实现步骤见 `docs/WAVE_DEVELOPMENT.md`；逐步操作和核验记录见 `docs/REFACTOR_LOG.md`。
