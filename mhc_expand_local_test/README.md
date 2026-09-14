# MhcExpand 本地辅助工具

完整的工程交接、WSL/CANN 8.5.0 环境搭建、编译、测试和提交说明见仓库根目录
[`README.md`](../README.md)。本目录不属于 CANNJudge 提交内容。

继续新一轮优化前，必须先读根目录
[`OPTIMIZATION_WORKFLOW.md`](../OPTIMIZATION_WORKFLOW.md)，按其中的实验命名、
fresh build、分层验证、打包校验和平台记录流程执行。

## 工具说明

- `compile_cann85.sh`：使用 CANN 8.5.0、`ascend910b` 配置依次构建 `binary`
  和 `install`，并保存 `build.log`。
- `package_submission.sh`：按原模板的 `code/` 结构，只打包 `B_easy` 中 7 个
  提交文件。
- `reference.py`：实现 forward/backward 的 PyTorch golden，并可生成原始数据。
- `test_reference.py`：验证 FP16/BF16、边界 shape 和非对齐 shape 的参考语义。

## 快速使用

从仓库根目录执行：

```bash
cd /home/sssafridi/kernel_optimization
source /home/sssafridi/miniconda3/etc/profile.d/conda.sh
conda activate cann85
MHC_EXP_ID=e3_forward_tile_16384
bash mhc_expand_local_test/compile_cann85.sh \
  "mhc_expand_local_test/build_${MHC_EXP_ID}"
```

运行 Python 参考测试：

```bash
cd /home/sssafridi/kernel_optimization/mhc_expand_local_test
python -m unittest -v test_reference.py
```

生成提交包：

```bash
cd /home/sssafridi/kernel_optimization
MHC_EXP_ID=e3_forward_tile_16384
bash mhc_expand_local_test/package_submission.sh \
  "mhc_expand_local_test/mhc_expand_${MHC_EXP_ID}.zip"
```

注意：Python 测试没有执行 Ascend Kernel；性能分数只能在 Ascend 910B 或
CANNJudge 上取得。
