# 日志分析报告
## 实验配置摘要
- dataset: RGB-D
- warmup_epochs: 20
- cross_warmup_epochs: 50

## 规则审计统计
- ERROR: 0
- WARN: 23
- INFO: 50

## 最优 vs 最后
- Best epoch=26 ACC=0.5424
- Last epoch=100 ACC=0.5017

## 阶段统计（均值）
- phase0_warmup: ACC=0.4512, L_cross_ratio=0.0000, FN_ratio=0.0407
- phase1_transition: ACC=0.5331, L_cross_ratio=0.0000, FN_ratio=0.0198
- phase2_cross: ACC=0.5183, L_cross_ratio=0.7387, FN_ratio=0.0140

## 机制相关性
- corr_FNratio_fnPairShare: 1.0000
- corr_HNratio_hnPairShare: 1.0000
- corr_deltaPost_Sp50: -0.8521
- corr_ACC_simNegP99: 0.8260
- corr_ACC_mgapP10: -0.1620
- corr_ACC_Lcross: 0.1764

## 主要告警样本（前30条）
- [WARN] epoch=3 route_instability: label_flip=0.981, stab_rate=0.019
- [WARN] epoch=5 route_instability: label_flip=0.973, stab_rate=0.027
- [WARN] epoch=13 route_instability: label_flip=0.995, stab_rate=0.005
- [WARN] epoch=18 route_instability: label_flip=0.995, stab_rate=0.005
- [WARN] epoch=20 route_instability: label_flip=0.998, stab_rate=0.002
- [WARN] epoch=21 route_instability: label_flip=0.998, stab_rate=0.002
- [WARN] epoch=26 route_instability: label_flip=0.995, stab_rate=0.005
- [WARN] epoch=27 route_instability: label_flip=0.998, stab_rate=0.002
- [WARN] epoch=29 route_instability: label_flip=0.983, stab_rate=0.017
- [WARN] epoch=33 route_instability: label_flip=0.998, stab_rate=0.002
- [WARN] epoch=48 route_instability: label_flip=0.972, stab_rate=0.028
- [WARN] epoch=53 route_instability: label_flip=0.999, stab_rate=0.001
- [WARN] epoch=57 route_instability: label_flip=0.994, stab_rate=0.006
- [WARN] epoch=62 route_instability: label_flip=0.996, stab_rate=0.004
- [WARN] epoch=65 route_instability: label_flip=0.975, stab_rate=0.025
- [WARN] epoch=74 route_instability: label_flip=0.962, stab_rate=0.038
- [WARN] epoch=76 route_instability: label_flip=1.000, stab_rate=0.000
- [WARN] epoch=77 route_instability: label_flip=0.999, stab_rate=0.001
- [WARN] epoch=83 tiny_cluster: min_cluster=9.0
- [WARN] epoch=84 tiny_cluster: min_cluster=9.0
- [WARN] epoch=88 route_instability: label_flip=0.986, stab_rate=0.014
- [WARN] epoch=92 route_instability: label_flip=1.000, stab_rate=0.000
- [WARN] epoch=96 route_instability: label_flip=1.000, stab_rate=0.000
