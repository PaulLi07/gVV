# 模块化等价重构工作日志

工作分支：`refactor/modular-architecture`

规范仓库：`/besfs10/groups/psip/psipgroup/user/liyuhong/GVV/analysis/pwa/ctpwa/Release/gVV`

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
- 修复后 Job `20730` 在 `gpu041` 上运行 2 分 50 秒并以 `COMPLETED 0:0`
  结束。10 个起点的最优解仍为 start 8 / seed `20260823`；新 NLL
  `-37530.4571965`，旧基线 `-37530.4571872`，差 `-9.3e-6`。12 个参数
  的最大绝对差为 `9.23e-4`，在 Minuit 数值收敛波动范围内；MIGRAD、
  HESSE 和 covariance status 均通过，EDM 为 `2.74e-5`。
- 四个 tag 输出全部存在：结果 TXT 含 12 个参数，covariance 为规则
  `12x12`，projection ROOT 为可读文件。ROOT 内含 `MC/data/bg`、
  `component_map/group_map/metadata` 六棵树；metadata 给出 7 Terms、
  2 JPC groups、23561 data、177374 normalization MC、有效产额 17358，
  分量闭合残差 `3.29e-15`。
- 仓库内残留的 209 MB 旧 `nominal/` 仅含生成二进制和历史输出，已移至
  可恢复临时备份 `/tmp/gvv_v1_legacy_artifacts_20260817_2116`，使规范
  项目树不再同时展示旧布局。

### 最终审计

- 数值回归后再次执行 `make clean && make -j2 tests fit && make check`，
  从零构建通过，9/9 测试通过。
- `framework/` 对 `process/` 的 include 扫描为空；Git 跟踪文件中没有
  ROOT、可执行文件、对象文件、批处理日志或生成图。
- 最终工作树 clean；本轮主要提交依次为 `3fd30f5`、`c54c7e7`、
  `e878776`、`e71ae38` 和 `941c3f3`。分支未推送远端。

## 项目更名与逐文件审查（2026-08-17）

- 确认 `Release/gVV` 不存在且原工作树 clean 后，把唯一仓库目录从
  `Release/gVV_v1` 更名为 `Release/gVV`；没有复制第二份仓库，Git 历史、
  分支和 remote 均保持不变。README、活动配置说明和规范路径同步改名；
  日志中的旧名称只作为历史记录保留。
- 逐文件审查 `app/`、`framework/`、`process/`、`config/`、`tests/`、
  `Makefile` 与 `submit.sh`；`postfit/` 仅检查更名影响，未修改其实现。
  所有生产源码和测试源码均补充文件级职责注释，关键公式、布局、状态机、
  配置编译和 Slurm 两阶段入口附近补充维护说明；架构文档新增代码阅读索引。
- 修复审查中发现的问题：`GVV_DATA_DIR` 指向仓库内 `RootSet/`，CUDA 动态库
  路径改为实际存在的 `lib64`；Makefile 生成并读取 `.d` 头文件依赖且不再用
  `-w` 隐藏告警；传播子描述只保留通用数值，GVV 的拟合策略回到过程层；
  传播子参数和 Term dynamics 拒绝多余/拼错字段；补充 Minuit 参数、样本状态、
  ROOT 分支读取、CUDA 模型状态和 projection 写入检查。
- 从零执行 `make clean && make -j2 tests fit && make check`，9/9 通过。
  工程源码无编译告警；仅 ROOT 6.32.02 外部 `TStorage.h` 产生两条
  `-Wattributes` 告警。自动依赖确认能从 `Scalar00.cuh` 追踪到
  `WaveRegistry.o` 和 `TermEvaluator.o`。
- GPU 回归 Job `20795` 在 `gpu041` 上以 `COMPLETED 0:0` 结束，用时
  2 分 41 秒。最优解仍为 start 8 / seed `20260823`，NLL
  `-37530.4571965`，与更名前回归完全一致；MIGRAD/HESSE 为 0，covariance
  status 为 3，EDM `2.73749e-5`。结果含 12 参数和 `12x12` covariance。
- projection ROOT 六棵树均可读：MC/data/bg 条目分别为
  177374/23561/27820，component/group map 为 7/2；有效产额 17358，分量
  闭合残差 `3.29041e-15`。framework 对 process 的 include/过程词扫描为空，
  `postfit/` diff 为空，生成物仍由 Git 忽略。

## 架构收尾：依赖与检查归位（2026-08-17）

- 把 `HBARC_GEV_FM` 和默认障碍半径归还 `BarrierFactor.cuh`，使
  `framework/tensors/` 不再反向包含 `framework/dynamics/`；框架物理工具
  恢复为 `math <- tensors <- dynamics <- process` 的单向依赖。
- 将 `FitState.cu` 的参数应用和物理结果明细合入 `ParameterMapping.cu`，删除
  单独对象和构建规则。模型到 Minuit vector 的布局、应用与序列化现在集中在
  一个模块中。
- 保留 JSON、ROOT/CUDA I/O、`Prepare()` 生命周期和似然数值边界检查；删除
  编译模型后的布局复查、每次参数写回时的索引/策略复查，以及组合 CUDA 入口
  已检查尺寸后对子阶段的重复检查。传播子不再静默截断负的 S/D 比例。
- `FitLikelihood` 与 `WaveRegistry` 的当前职责和后续可收窄边界补入架构文档。
  本轮验证不提交 Slurm、HTCondor 或 GPU 作业；集群运行验证由用户执行。
- `git diff --check`、Shell/JSON 静态检查和依赖方向扫描通过；执行
  `make clean && make -j2 tests fit && make check`，干净编译及 9/9 单元测试
  通过。未运行 `Fit.exe`。编译输出只有既有的 sm70 目标弃用提示与 ROOT
  `TStorage.h` 外部头文件告警，没有项目源码告警。

## 架构收尾：振幅与投影职责拆分（2026-08-17）

- 新建 `process/ProcessAmplitude.cuh`，从 `WaveRegistry.cuh` 整体迁移光子
  偏振投影、Wave 两两缩并和 `F_ij` 组装。注册表现在只拥有 Wave 枚举、设备
  dispatch、主机注册元数据和编译模型类型，`TermEvaluator` 单向依赖振幅接口。
- 新建 `process/ProjectionWriter.h/.cu`，集中保存原 projection ROOT 的树结构、
  派生运动学、权重分解、映射表和 metadata。`FitLikelihood` 只保留样本准备、
  GPU 模型同步、强度计算、MC 归一化和有符号似然，并通过窄接口向 writer 提供
  样本与指定耦合下的 normalization-MC 强度。
- 保持现有 projection 文件名、树名、branch 名、权重定义和 metadata 语义不变；
  `app/Fit.cu` 只把最终拟合状态交给独立 writer。下游脚本仍不在本轮修改范围。
- 在固定节点执行 `make clean && make -j2 tests fit && make check`：新 writer
  独立编译并链接进 `Fit.exe`，9/9 单元测试通过。Shell/JSON、补丁格式、框架
  依赖方向、模块职责关键词、自动头文件依赖及 `postfit/`/配置零改动审计通过。
  未运行 `Fit.exe`，未提交任何 Slurm、HTCondor 或 GPU 作业；projection 的
  运行时文件核验留给用户执行。
