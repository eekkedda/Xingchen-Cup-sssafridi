# 910B 真机 Runbook(CANN 9.0 环境)

> 目的:在真机上完成本地做不到的三件事——**真机计时分解**(小点 2.3μs 之谜)、
> **golden 输出比对**(mskpp 在真机无 camodel 挂起问题)、**tiling A/B 实测**。
> 注意:比赛评测用 CANN 8.5;真机 9.0 的结论用于**方向判断**,最终仍以平台为准。

## 1. 环境自检(登录后先跑)

```bash
npu-smi info                       # 确认 910B 在位
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg 2>/dev/null \
  || find /usr/local/Ascend ~/Ascend -maxdepth 3 -name 'version*' 2>/dev/null
source /usr/local/Ascend/ascend-toolkit/set_env.sh   # 按实际路径
which cmake && cmake --version      # >= 3.16
```

## 2. 编译(用仓库内脚本,任何 CANN 版本可跑)

```bash
cd ~/kernel_optimization/mhc_expand_local_test
bash compile_remote.sh ./build_910b
# 产物: build_910b/op_kernel/ascendc_kernels/binary/ascend910b/mhc_expand/*.o
```

## 3. 真机 kernel 计时(msprof,真机模式非模拟)

对小型 shape 用 ACLNN 单算子调用计时(msprof op 需要可执行入口,推荐直接用
mskpp 流程,见第 4 节,它内部走 aclrt 真机 launch 并可 repeat 计时):

重点测(对应平台点 1/5 家族,假设 S=64/D=256/m=2 与 S=1/D=256/m=2,
以及 D=16 极小):
- E4G2 当前 tiling
- blockDim 扫描:1/4/12/32/48(mskpp 里 kernel[bd](...) 可任意 bd)
- 对照:点 1 平台 ~3.6μs,真机单 kernel 耗时若 ~1.5μs ⇒ 差值为下发/框架开销;
  若真机也 ~3.5μs ⇒ kernel 内部还有货,profile 指令流水找热点

## 4. mskpp 真机 golden 比对(打通本地安全网)

仓库脚本 `MhcExpand_Round2/simulator/run_mskpp_case.py` 在真机环境去掉
camodel 相关 LD_PRELOAD 即可(真机走 runtime 而非 libruntime_camodel):

```bash
cd ~/kernel_optimization/MhcExpand_Round2/simulator
# 修改 run_mskpp_case.py: BUILD 指向 build_910b(或保持 e4g2 构建目录名)
python3 run_mskpp_case.py fp16_forward_small     # 输出与 golden 逐位比对
python3 run_mskpp_case.py fp16_backward_small
python3 run_mskpp_case.py fp16_backward_m8
```
若 numpy 缺失: pip3 install numpy(或用系统 python)。g++ 缺失时把
~/.local/bin 的 g++ 软链方案换成本机包管理器安装。

## 5. tiling A/B 实测(模拟器结论的最终仲裁)

用 simulator 目录里的 fixture 生成器(需 numpy)重建 tiling bin:
`gen_mid_balance_cases.py`、`gen_e6_cases.py`、`gen_e2_cases.py` 等,
把 config json 的 `mode: ca` 在真机流程下不需要(mskpp 直接 launch)。
优先重测:R1 vs R2(中规模)、分组 mode 0/1/2(反向)、blockDim 扫描。

## 6. 数据回传

结果(计时文本、o.bin、profile)用同样的传输通道反向拷回本地,
归档进 `MhcExpand_Round2/` 对应实验 JSON,保持单一事实源在本地仓库。

## 7. 红线提醒(真机上也不许做)

不按隐藏测试点编号/shape 指纹特化;不在 host 做数值计算;最终提交
必须过 CANN 8.5 评测——真机 9.0 编译产物不直接用于提交,只用其结论。
