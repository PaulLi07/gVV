# gVV v1.0.0-rc1 validation record

Date: 2026-08-16 (Asia/Shanghai)  
Node: `lxlogin005.ihep.ac.cn`  
Project: `/besfs10/groups/psip/psipgroup/user/liyuhong/GVV/analysis/pwa/ctpwa/Release/gVV_v1/nominal`

## Scope

This record covers the release-layout normalization, Slurm project-root fix,
plot-macro relocation, documentation update, build-time tests, and lightweight
runtime checks. No Fit/PostFit batch job was submitted during this validation.

## Passed checks

1. Shell syntax:

   ```bash
   bash -n scripts/Sub.sh scripts/subgpu.sh scripts/Sub_postfit.sh \
     scripts/subpostgpu.sh scripts/draw.sh config/gvv_env.sh
   ```

2. Environment loading:

   - CUDA path: `/usr/local/cuda-12`
   - nvcc: CUDA 12.9 build
   - ROOT: 6.32.02

3. Test build:

   ```bash
   make tests -j2
   ```

   All four test executables were produced in `tests/bin/`. nvcc emitted only
   the expected future-deprecation warning for the configured `sm_70` target.

4. Unit/regression and PostFit algebra checks:

   ```text
   Dynamics tests passed
   GVV amplitude device path compiled
   GVV model and propagator tests passed
   GVV fit-parameter tests passed
   GVV PostFit algebra self-test passed
   ```

5. Slurm spool-path simulation:

   Copies of `subgpu.sh` and `subpostgpu.sh` were executed from `/tmp`, while
   `GVV_PROJECT_ROOT` pointed to `nominal/`. Both loaded the correct project
   `config/gvv_env.sh` and reached their usage checks instead of resolving the
   project as `/var/spool/slurm/spool`.

6. Plot macro loading:

   ROOT loaded all five macros from `scripts/plot/` without include or syntax
   errors. Generated figures are configured for `results/plot/`.

7. Documentation:

   The architecture HTML passed structural validation, 31-unit coverage, anchor
   validation, and local-link validation.

## Remaining release gates

- Submit one explicit single-start GPU smoke fit:

  ```bash
  ./scripts/Sub.sh 1 20260815
  ```

- Verify scheduler state, Minuit/HESSE status, covariance status, EDM, parameter
  boundaries, and projection closure from the same JobID.
- Run `scripts/draw.sh` on the new `results/projection0.root` and inspect all
  files under `results/plot/`.
- When the matching all-generated truth MC is available, run PostFit and verify
  fraction/interference/efficiency closure.
