# GVV v1 构建、提交和后处理说明

## 1. 进入项目环境

在 `nominal/` 目录执行：

```bash
source config/gvv_env.sh
```

该配置只管理本项目使用的 CUDA、ROOT、`RootSet` 和 PATH，不修改全局
`setup_ctpwa`。`Fit.exe`、`PostFit.exe` 和所有 worker 脚本都会再次使用这份配置。

## 2. 构建

```bash
make clean
make -j4
make tests -j4
```

产物位置：

- `bin/Fit.exe`、`bin/PostFit.exe`；
- `build/obj/*.o`；
- `tests/bin/*.exe`。

正式构建建议保存日志：

```bash
make clean
make -j4 2>&1 | tee runlog/build.log
make tests -j4 2>&1 | tee runlog/build-tests.log
```

## 3. Fit 提交

推荐使用 `scripts/` 中的 shell 入口：

```bash
cd scripts
./Sub.sh
```

从 `nominal/` 调用也可以：

```bash
./scripts/Sub.sh
```

`Sub.sh [n_starts] [base_seed]` 默认使用 10 个起点和种子
`20260815`。例如显式选择 20 个起点：

```bash
./Sub.sh 20 20260815
```

脚本会通过 `--chdir` 和 `GVV_PROJECT_ROOT` 把真实项目根传给 Slurm，避免 worker
被复制成 `/var/spool/.../slurm_script` 后错误地从 `BASH_SOURCE` 推断路径。
`subgpu.sh` 申请一个 A100 并运行 `bin/Fit.exe`。默认输入来自 `../RootSet/`；也可
设置 `GVV_DATA_DIR` 覆盖输入目录。Fit 输出写入 `results/`，Slurm 输出写入
`runlog/`。

## 4. PostFit 提交

Fit 完成且 truth MC 准备好后：

从 `nominal/` 执行：

```bash
./scripts/Sub_postfit.sh \
  "$GVV_DATA_DIR/truth_mc.root" \
  "$GVV_DATA_DIR/normalization_mc.root"
```

truth MC 是同一批 PHSP 生成事件的全生成样本，normalization MC 是其通过选择后的
子样本。PostFit 计算 fit fraction、干涉、效率及其协方差传播，不参与拟合 NLL。

## 5. 绘图

```bash
./scripts/draw.sh results/projection0.root
```

不提供参数时使用 `results/projection0.root`。`draw.sh` 调用 `scripts/plot/` 中的
五个 ROOT 宏，输出 PDF/EPS 到 `results/plot/`。

## 6. 快速验证

```bash
cd nominal
source config/gvv_env.sh
./tests/bin/test_dynamics.exe
./tests/bin/test_gvv_amplitude.exe
./tests/bin/test_gvv_model.exe
./tests/bin/test_gvv_fit_parameters.exe
./bin/PostFit.exe --self-test
bash -n scripts/*.sh config/gvv_env.sh
```

`tests/` 保存四个 CUDA/C++ 测试源文件和 fit-result 解析 fixtures；`make tests`
只负责编译到 `tests/bin/`。已有真实拟合结果时，可额外运行：

```bash
./tests/bin/test_gvv_fit_parameters.exe results/fit_result-initial.txt
```
