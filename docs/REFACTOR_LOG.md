# 模块化等价重构工作日志

工作分支：`refactor/modular-architecture`

规范仓库：`/besfs10/groups/psip/psipgroup/user/liyuhong/GVV/analysis/pwa/ctpwa/Release/gVV_v1`

固定登录节点：`lxlogin005.ihep.ac.cn`；持久会话：`gvv_general`。

## 范围与不变量

本次只做架构等价重构：不新增或删除物理共振态，不实现新的 `2++` Wave，不改变现有传播子公式、Wave 张量公式、omega 宽度表、MC 归一化或二维边带系数。拟合后绘图系统不在本次范围；用户已调整的投影绘图样式原样保存在 `postfit/`。

明确删除旧拟合结果读取兼容层。新的 Fit 只写当前输出协议，不读取旧 TXT，也不生成模型/fit 配置快照。

## 仓库整理

- 早期隔离工作曾使用 `gVV_v1_refactor_modular` worktree；按用户“一份仓库”的要求，其内容已合回规范路径并删除临时 worktree。
- 接受并保留用户对旧 `gVV_v1` 的绘图样式修改。
- 从 `5033604` 继续工作，始终使用同一规范仓库和同一分支；没有创建第二个项目仓库。
- 提交 `3fd30f5`：建立根级 `framework/`、`process/`、`app/`、`config/`、`tests/` 与 `postfit/` 边界。

## 阶段一：数学、动力学与 Wave 边界

- 从旧 `nominal/` 提取通用四矢量、复数、度规、Levi-Civita、张量、自旋投影、轨道张量和障碍因子。
- 提取通用二体运动学与传播子注册/统一求值接口；GVV omega 常量留在过程层。
- 把三个现有完整 Wave 分为：
  - `process/waves/Scalar00.cuh`
  - `process/waves/Scalar22.cuh`
  - `process/waves/Pseudoscalar11.cuh`
- 建立 Wave 的唯一主机注册点和唯一设备 dispatch。
- Resonance、Wave、Term 和自由耦合全部使用运行时 vector；生产路径无固定总共振态数、Term 数或自由参数数。
- 首轮真实 CUDA/ROOT 编译通过，七项单元检查全部通过。

## 阶段二：拟合与输出边界

- `framework/fit/FitEngine`：过程无关的多起点 TMinuit、随机起点、收敛筛选和最优解选择。
- `framework/fit/FitConfig`：严格读取唯一 `config/fit.json`。
- `framework/fit/FitOutput`：统一机器可读参数行和协方差输出。
- `framework/likelihood/Likelihood.h`：过程无关的 MC 归一化与有符号无分箱似然算术。
- `process/FitLikelihood`：只负责编排 GVV 样本、F 矩阵、omega 宽度表、GPU 状态和过程 projection。
- `process/ParameterMapping`：唯一的模型到 Minuit 参数映射；Minuit 层不认识 Resonance/Wave。
- 删除旧结果解析器、旧 covariance 读取兼容接口及两份旧格式 fixture。
- `app/Fit.cu` 缩为配置加载、过程对象组装、拟合调用和输出调用。
- 新增配置契约测试；真实 CUDA 12 / ROOT 6.32.02 完整链接 `bin/Fit.exe` 成功，`make check` 八项检查全部通过。

## 阶段三：运行接口与文档

- 用根目录唯一 `submit.sh` 取代旧提交端/worker 两脚本；使用项目既有授权组合 `gpupwa/gpupwa/pwadedicate` 和一张 A100。
- `output.tag` 同时决定 TXT、covariance、ROOT projection 和 Slurm log；同名直接覆盖。
- 重写架构、配置和新 Wave 开发文档，明确未来更换末态时复用 framework、替换 process 的边界。
- 原绘图和拟合后处理代码集中在 `postfit/`，此次不迁移其旧结果读取接口，也不纳入默认构建。

## 核验记录

截至架构编译阶段：

- `bash -n submit.sh`：通过；
- `make -j2 tests fit`：通过；
- `make check`：9/9 通过（含通用 Minuit 二次函数收敛测试）；
- `git diff --check`：通过；
- `Fit.exe`：在 IHEP CUDA/ROOT 环境完整链接成功；
- 未把 ROOT、二进制、构建目录或历史拟合结果加入 Git。

最终 GPU 数值等价拟合、输出文件检查、Git 审计和对应提交号在实际核验完成后追加到本节。

### GPU 回归过程

- 首次提交 Job `20729` 在应用启动前以 `ExitCode 4:0` 失败：Slurm
  worker 从 spool 脚本路径错误推导项目根目录，找不到 `config/model.json`。
  修复为优先使用提交端导出的 `GVV_PROJECT_ROOT` 后重新核验；该失败不曾
  运行拟合或写入数值结果。
