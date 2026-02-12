#!/usr/bin/env python3
"""
实验日志统计与可视化工具（MVSIB 专用）

功能目标：
1) 解析 train.py 产出的 Args / METRIC / ROUTE / DISTR / Final/Best 段落；
2) 做硬不变量 + 软规则审计，输出 ERROR/WARN/INFO 分级报告；
3) 进行阶段统计、机制闭环分析、loss 占比分析；
4) 生成面向实验复盘的可视化图表与结构化数据文件。
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt


KV_RE = re.compile(r"([A-Za-z0-9_]+)=([^\s]+)")


@dataclass
class RuleEvent:
    """规则事件。

    level: ERROR/WARN/INFO
    rule: 规则名称
    epoch: 触发的 epoch（如无则 -1）
    detail: 详细说明
    """

    level: str
    rule: str
    epoch: int
    detail: str


@dataclass
class ParsedLog:
    """日志结构化结果。

    - args_dict: Args 字段
    - metric_rows / route_rows / distr_rows: 各条目（按 epoch）
    - final_summary: 结尾摘要字段
    """

    args_dict: Dict[str, Any]
    metric_rows: List[Dict[str, Any]]
    route_rows: List[Dict[str, Any]]
    distr_rows: List[Dict[str, Any]]
    final_summary: Dict[str, Any]


def _to_num(v: str) -> Any:
    """把日志值转为数字或字符串。

    目的：统一处理 int/float/布尔/字符串，便于后续统计。
    结果说明：若无法转数字则保留原字符串，保证解析不崩溃。
    """

    lv = v.lower()
    if lv in {"true", "false"}:
        return lv == "true"
    if lv == "nan":
        return float("nan")
    try:
        if any(ch in v for ch in ".eE"):
            return float(v)
        return int(v)
    except Exception:
        return v


def parse_kv_segment(seg: str) -> Dict[str, Any]:
    """解析形如 key=value 的平铺字段串。

    目的：解析 METRIC/ROUTE/DISTR 行主体。
    结果说明：返回字典，键为字段名，值为数字或字符串。
    """

    out = {}
    for k, v in KV_RE.findall(seg):
        out[k] = _to_num(v.strip().rstrip(','))
    return out


def parse_args_line(line: str) -> Dict[str, Any]:
    """解析 Args: Namespace(...) 行。

    目的：提取实验配置，供阶段切分和报表标题使用。
    结果说明：若解析失败返回空字典，不阻塞主流程。
    """

    m = re.search(r"Args:\s*Namespace\((.*)\)$", line)
    if not m:
        return {}
    inner = m.group(1)
    # 将 a=1,b='x' 包装成 dict 字面量再 ast 解析
    try:
        py_expr = "{" + re.sub(r"(\w+)=", r"'\1':", inner) + "}"
        return ast.literal_eval(py_expr)
    except Exception:
        return {}


def parse_log_file(log_path: Path) -> ParsedLog:
    """解析单个日志文件。

    目的：把原始日志映射为结构化数据，供统计/审计/作图复用。
    结果说明：返回 ParsedLog；缺失段不会抛异常，尽可能容错。
    """

    args_dict: Dict[str, Any] = {}
    metric_rows: List[Dict[str, Any]] = []
    route_rows: List[Dict[str, Any]] = []
    distr_rows: List[Dict[str, Any]] = []
    final_summary: Dict[str, Any] = {}

    lines = log_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    for ln in lines:
        if "Args: Namespace(" in ln:
            args_dict = parse_args_line(ln)
        elif "METRIC:" in ln:
            seg = ln.split("METRIC:", 1)[1].strip()
            d = parse_kv_segment(seg)
            if "epoch" in d:
                metric_rows.append(d)
        elif "ROUTE:" in ln:
            seg = ln.split("ROUTE:", 1)[1].strip()
            d = parse_kv_segment(seg)
            if "epoch" in d:
                route_rows.append(d)
        elif "DISTR:" in ln:
            seg = ln.split("DISTR:", 1)[1].strip()
            d = parse_kv_segment(seg)
            if "epoch" in d:
                distr_rows.append(d)
        elif "Final Evaluation (Last Epoch)" in ln:
            final_summary["final_marker"] = True
        elif "Best Evaluation" in ln:
            final_summary["best_line"] = ln

    # 按 epoch 排序，避免乱序影响分析
    metric_rows.sort(key=lambda x: x.get("epoch", 0))
    route_rows.sort(key=lambda x: x.get("epoch", 0))
    distr_rows.sort(key=lambda x: x.get("epoch", 0))

    return ParsedLog(args_dict, metric_rows, route_rows, distr_rows, final_summary)


def merge_by_epoch(*rows_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按 epoch 外连接合并多类行。

    目的：统一得到一个“每 epoch 一行”的宽表，便于规则和关联分析。
    结果说明：缺字段会保留为缺失，不会强行填错值。
    """

    by_epoch: Dict[int, Dict[str, Any]] = {}
    for rows in rows_list:
        for r in rows:
            ep = int(r.get("epoch", -1))
            if ep < 0:
                continue
            by_epoch.setdefault(ep, {"epoch": ep}).update(r)
    return [by_epoch[k] for k in sorted(by_epoch.keys())]


def phase_ranges(args_dict: Dict[str, Any], max_epoch: int) -> List[Tuple[str, int, int]]:
    """基于训练里程碑定义阶段。

    目的：和训练机制对齐（warmup / cross_warmup），避免任意切段。
    结果说明：输出 phase0/1/2 的 epoch 区间，供图表和统计复用。
    """

    w = int(args_dict.get("warmup_epochs", 20))
    cw = int(args_dict.get("cross_warmup_epochs", 50))
    w = min(max(w, 1), max_epoch)
    cw = min(max(cw, w), max_epoch)
    return [
        ("phase0_warmup", 1, w),
        ("phase1_transition", w + 1, cw),
        ("phase2_cross", cw + 1, max_epoch),
    ]


def _safe(v: Any, default=np.nan) -> float:
    """安全取 float 值，失败回 NaN。"""

    try:
        return float(v)
    except Exception:
        return float(default)


def validate_rules(rows: List[Dict[str, Any]], eps: float = 5e-3) -> List[RuleEvent]:
    """执行硬不变量与软规则审计。

    目的：把“看图判断”转成可复现的实验审计规则。
    结果说明：返回 RuleEvent 列表（ERROR/WARN/INFO），并记录触发 epoch。
    """

    events: List[RuleEvent] = []

    for r in rows:
        ep = int(r.get("epoch", -1))
        fnr = _safe(r.get("FN_ratio", r.get("fn_ratio", np.nan)))
        hnr = _safe(r.get("HN_ratio", r.get("hn_ratio", np.nan)))
        fnc = _safe(r.get("FN_count", np.nan))
        hnc = _safe(r.get("HN_count", np.nan))
        neg = _safe(r.get("neg_count", np.nan))
        safe_neg = _safe(r.get("safe_neg_count", np.nan))
        cand = _safe(r.get("candidate_neg_size", np.nan))
        filt = _safe(r.get("neg_after_filter_size", np.nan))
        used = _safe(r.get("neg_used_in_loss_size", np.nan))
        u_size = _safe(r.get("U_size", np.nan))
        fshare = _safe(r.get("fn_pair_share", np.nan))
        hshare = _safe(r.get("hn_pair_share", np.nan))
        label_flip = _safe(r.get("label_flip", np.nan))
        stab_rate = _safe(r.get("stab_rate", np.nan))
        w_hit = _safe(r.get("w_hit_min_ratio", np.nan))
        empty_cluster = _safe(r.get("empty_cluster", np.nan))
        min_cluster = _safe(r.get("min_cluster", np.nan))

        # ===== 硬不变量 =====
        if not (0.0 - eps <= fnr <= 1.0 + eps):
            events.append(RuleEvent("ERROR", "fn_ratio_range", ep, f"FN_ratio={fnr}"))
        if not (0.0 - eps <= hnr <= 1.0 + eps):
            events.append(RuleEvent("ERROR", "hn_ratio_range", ep, f"HN_ratio={hnr}"))
        if not (math.isnan(cand) or math.isnan(filt) or cand + eps >= filt):
            events.append(RuleEvent("ERROR", "candidate_vs_filtered", ep, f"candidate={cand}, filtered={filt}"))
        if not (math.isnan(u_size) or u_size <= 0 or math.isnan(used) or used > 0):
            events.append(RuleEvent("ERROR", "neg_used_nonzero", ep, f"U_size={u_size}, used={used}"))

        # 比率-计数重算一致性（count-based share）
        if not (math.isnan(fnc) or math.isnan(neg) or neg <= 0):
            fnr_re = fnc / neg
            if not math.isnan(fnr) and abs(fnr - fnr_re) > 0.05:  # 容许不同口径，但偏差太大报警
                events.append(RuleEvent("WARN", "fn_ratio_recompute_gap", ep, f"FN_ratio={fnr:.4f}, FN_count/neg_count={fnr_re:.4f}"))
        if not (math.isnan(hnc) or math.isnan(safe_neg) or safe_neg <= 0):
            hnr_re = hnc / safe_neg
            if not math.isnan(hnr) and abs(hnr - hnr_re) > 0.05:
                events.append(RuleEvent("WARN", "hn_ratio_recompute_gap", ep, f"HN_ratio={hnr:.4f}, HN_count/safe_neg={hnr_re:.4f}"))

        # share 强一致（通常应接近）
        if not (math.isnan(fshare) or math.isnan(fnc) or math.isnan(neg) or neg <= 0):
            if abs(fshare - fnc / neg) > 0.02:
                events.append(RuleEvent("ERROR", "fn_pair_share_consistency", ep, f"fn_pair_share={fshare:.4f}, FN_count/neg_count={fnc/neg:.4f}"))
        if not (math.isnan(hshare) or math.isnan(hnc) or math.isnan(safe_neg) or safe_neg <= 0):
            if abs(hshare - hnc / safe_neg) > 0.02:
                events.append(RuleEvent("ERROR", "hn_pair_share_consistency", ep, f"hn_pair_share={hshare:.4f}, HN_count/safe_neg={hnc/safe_neg:.4f}"))

        # ===== 软规则 =====
        if (not math.isnan(label_flip) and not math.isnan(stab_rate) and label_flip > 0.95 and stab_rate < 0.05):
            events.append(RuleEvent("WARN", "route_instability", ep, f"label_flip={label_flip:.3f}, stab_rate={stab_rate:.3f}"))
        if not math.isnan(w_hit) and w_hit == 0.0:
            events.append(RuleEvent("INFO", "w_hit_min_zero", ep, "w_hit_min_ratio=0"))
        if not math.isnan(empty_cluster) and empty_cluster > 0:
            events.append(RuleEvent("WARN", "empty_cluster", ep, f"empty_cluster={empty_cluster}"))
        if not math.isnan(min_cluster) and min_cluster < 10:
            events.append(RuleEvent("WARN", "tiny_cluster", ep, f"min_cluster={min_cluster}"))

    return events


def summarize_phase(rows: List[Dict[str, Any]], phases: List[Tuple[str, int, int]]) -> Dict[str, Dict[str, float]]:
    """分阶段汇总关键指标均值。

    目的：回答“在哪个阶段发生了机制变化/性能退化”。
    结果说明：输出 phase -> 指标均值字典，可用于报告表格。
    """

    keys = ["ACC", "F1", "L_total", "L_cross", "L_feat", "FN_ratio", "HN_ratio", "gate"]
    out: Dict[str, Dict[str, float]] = {}
    for name, lo, hi in phases:
        sub = [r for r in rows if lo <= int(r.get("epoch", -1)) <= hi]
        stats = {}
        for k in keys:
            vals = np.array([_safe(r.get(k, np.nan)) for r in sub], dtype=float)
            vals = vals[np.isfinite(vals)]
            stats[k] = float(np.mean(vals)) if vals.size else float("nan")
        if math.isfinite(stats.get("L_total", float("nan"))) and stats["L_total"] > 0:
            stats["L_cross_ratio"] = stats.get("L_cross", float("nan")) / stats["L_total"]
            stats["L_feat_ratio"] = stats.get("L_feat", float("nan")) / stats["L_total"]
        out[name] = stats
    return out


def corr(a: np.ndarray, b: np.ndarray) -> float:
    """皮尔逊相关。

    目的：量化机制量与性能之间的同步关系。
    结果说明：返回 [-1,1]，绝对值越大关系越强（不代表因果）。
    """

    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def mechanism_analysis(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """机制闭环分析（ROUTE ↔ DISTR ↔ METRIC）。

    目的：解释性能变化对应的路由/分布机制变化，而不是只看单一曲线。
    结果说明：输出关键相关性，辅助定位退化阶段的机制原因。
    """

    arr = lambda key: np.array([_safe(r.get(key, np.nan)) for r in rows], dtype=float)
    return {
        "corr_FNratio_fnPairShare": corr(arr("FN_ratio"), arr("fn_pair_share")),
        "corr_HNratio_hnPairShare": corr(arr("HN_ratio"), arr("hn_pair_share")),
        "corr_deltaPost_Sp50": corr(arr("delta_post"), arr("S_p50")),
        "corr_ACC_simNegP99": corr(arr("ACC"), arr("sim_neg_p99")),
        "corr_ACC_mgapP10": corr(arr("ACC"), arr("m_gap_p10")),
        "corr_ACC_Lcross": corr(arr("ACC"), arr("L_cross")),
    }


def save_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    """保存宽表 CSV。

    目的：便于后续 pandas/Jupyter 再分析和论文图复现。
    结果说明：字段并集写出，不丢信息。
    """

    keys = sorted({k for r in rows for k in r.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_dashboard(rows: List[Dict[str, Any]], phases: List[Tuple[str, int, int]], out_png: Path) -> None:
    """阶段标注总览图（核心图）。

    计算/可视化内容：
    - 子图1：ACC/F1 曲线；
    - 子图2：L_total 与 L_cross/L_total；
    - 子图3：FN/HN ratio；
    - 子图4：candidate/filter/used 负样本规模；
    - 子图5：u_p50/gamma_p50/S_p50；
    - 子图6：route_count_inconsistent 与 empty_cluster。

    方法目的：
    - 用一张图概览“性能-损失-路由-分布-一致性”闭环；
    - 标注阶段边界后更容易定位机制切换点（warmup/cross_warmup）。

    结果说明：
    - 若 phase2 中 L_cross 占比上升且 ACC 下滑，提示一致性项可能后期主导；
    - 若 route inconsistency/empty cluster 抬升，提示训练统计或聚类退化风险。
    """

    ep = np.array([int(r.get("epoch", 0)) for r in rows])
    get = lambda k: np.array([_safe(r.get(k, np.nan)) for r in rows], dtype=float)

    acc, f1 = get("ACC"), get("F1")
    ltot, lcross = get("L_total"), get("L_cross")
    fnr, hnr = get("FN_ratio"), get("HN_ratio")
    cand, flt, used = get("candidate_neg_size"), get("neg_after_filter_size"), get("neg_used_in_loss_size")
    up50, gp50, sp50 = get("u_p50"), get("gamma_p50"), get("S_p50")
    inc, empty = get("route_count_inconsistent"), get("empty_cluster")

    ratio = np.divide(lcross, ltot, out=np.full_like(lcross, np.nan), where=np.isfinite(ltot) & (ltot > 0))

    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    axes = axes.reshape(3, 2)

    axes[0, 0].plot(ep, acc, label="ACC")
    axes[0, 0].plot(ep, f1, label="F1")
    axes[0, 0].set_title("性能曲线")
    axes[0, 0].legend()

    axes[0, 1].plot(ep, ltot, label="L_total")
    ax2 = axes[0, 1].twinx()
    ax2.plot(ep, ratio, color="tab:red", label="L_cross/L_total")
    axes[0, 1].set_title("总损失与跨视图占比")

    axes[1, 0].plot(ep, fnr, label="FN_ratio")
    axes[1, 0].plot(ep, hnr, label="HN_ratio")
    axes[1, 0].set_title("ROUTE 比率")
    axes[1, 0].legend()

    axes[1, 1].plot(ep, cand, label="candidate")
    axes[1, 1].plot(ep, flt, label="after_filter")
    axes[1, 1].plot(ep, used, label="used_in_loss")
    axes[1, 1].set_title("负样本链路规模")
    axes[1, 1].legend()

    axes[2, 0].plot(ep, up50, label="u_p50")
    axes[2, 0].plot(ep, gp50, label="gamma_p50")
    axes[2, 0].plot(ep, sp50, label="S_p50")
    axes[2, 0].set_title("DISTR 机制观测")
    axes[2, 0].legend()

    axes[2, 1].plot(ep, inc, label="route_inconsistent")
    axes[2, 1].plot(ep, empty, label="empty_cluster")
    axes[2, 1].set_title("一致性与聚类健康")
    axes[2, 1].legend()

    for _, lo, hi in phases:
        for ax in axes.flatten():
            if lo > 1:
                ax.axvline(lo, ls="--", lw=0.8, color="gray", alpha=0.6)
            ax.axvline(hi, ls=":", lw=0.8, color="gray", alpha=0.6)

    axes[2, 0].set_xlabel("epoch")
    axes[2, 1].set_xlabel("epoch")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def plot_loss_ratio(rows: List[Dict[str, Any]], out_png: Path) -> None:
    """绘制 loss 占比图。

    计算/可视化内容：L_cross/L_total、L_feat/L_total、L_cluster/L_total、L_uncert/L_total、L_hn/L_total。
    方法目的：识别“谁在主导优化”，比绝对值更利于解释退化。
    结果说明：某项长期占比异常抬升，通常代表该项主导梯度更新。
    """

    ep = np.array([int(r.get("epoch", 0)) for r in rows])
    get = lambda k: np.array([_safe(r.get(k, np.nan)) for r in rows], dtype=float)
    ltot = get("L_total")

    def ratio(k: str) -> np.ndarray:
        x = get(k)
        return np.divide(x, ltot, out=np.full_like(x, np.nan), where=np.isfinite(ltot) & (ltot > 0))

    plt.figure(figsize=(12, 6))
    for key in ["L_cross", "L_feat", "L_cluster", "L_uncert", "L_hn"]:
        plt.plot(ep, ratio(key), label=f"{key}/L_total")
    plt.title("Loss 占比曲线")
    plt.xlabel("epoch")
    plt.ylabel("ratio")
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()


def write_report_md(path: Path, args_dict: Dict[str, Any], phase_stat: Dict[str, Dict[str, float]],
                    mech: Dict[str, float], events: List[RuleEvent], rows: List[Dict[str, Any]]) -> None:
    """写 Markdown 总结。

    目的：输出可直接贴实验记录/PR 的文字结论。
    结果说明：包含规则分级、阶段统计、机制相关性与关键结论。
    """

    e_cnt = sum(1 for e in events if e.level == "ERROR")
    w_cnt = sum(1 for e in events if e.level == "WARN")
    i_cnt = sum(1 for e in events if e.level == "INFO")

    best_row = max(rows, key=lambda r: _safe(r.get("ACC", np.nan))) if rows else {}
    last_row = rows[-1] if rows else {}

    lines = []
    lines.append("# 日志分析报告\n")
    lines.append("## 实验配置摘要\n")
    lines.append(f"- dataset: {args_dict.get('dataset', 'unknown')}\n")
    lines.append(f"- warmup_epochs: {args_dict.get('warmup_epochs', 'N/A')}\n")
    lines.append(f"- cross_warmup_epochs: {args_dict.get('cross_warmup_epochs', 'N/A')}\n")

    lines.append("\n## 规则审计统计\n")
    lines.append(f"- ERROR: {e_cnt}\n- WARN: {w_cnt}\n- INFO: {i_cnt}\n")

    lines.append("\n## 最优 vs 最后\n")
    lines.append(f"- Best epoch={best_row.get('epoch')} ACC={_safe(best_row.get('ACC')):.4f}\n")
    lines.append(f"- Last epoch={last_row.get('epoch')} ACC={_safe(last_row.get('ACC')):.4f}\n")

    lines.append("\n## 阶段统计（均值）\n")
    for ph, st in phase_stat.items():
        lines.append(f"- {ph}: ACC={st.get('ACC', float('nan')):.4f}, L_cross_ratio={st.get('L_cross_ratio', float('nan')):.4f}, FN_ratio={st.get('FN_ratio', float('nan')):.4f}\n")

    lines.append("\n## 机制相关性\n")
    for k, v in mech.items():
        lines.append(f"- {k}: {v:.4f}\n")

    # 列出前若干严重问题
    bad = [e for e in events if e.level in {"ERROR", "WARN"}][:30]
    if bad:
        lines.append("\n## 主要告警样本（前30条）\n")
        for e in bad:
            lines.append(f"- [{e.level}] epoch={e.epoch} {e.rule}: {e.detail}\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


def save_events_json(path: Path, events: List[RuleEvent]) -> None:
    """保存规则事件 JSON。"""

    data = [e.__dict__ for e in events]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    """主入口。

    执行流程：
    1) 解析日志；
    2) 合并宽表；
    3) 规则审计；
    4) 阶段/机制统计；
    5) 导出 CSV/JSON/MD/PNG。
    """

    ap = argparse.ArgumentParser(description="Analyze MVSIB train logs")
    ap.add_argument("--log", required=True, help="log file path")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--strict-hard", action="store_true", help="exit 2 if ERROR exists")
    args = ap.parse_args()

    log_path = Path(args.log)
    out_dir = Path(args.out)

    parsed = parse_log_file(log_path)
    merged = merge_by_epoch(parsed.metric_rows, parsed.route_rows, parsed.distr_rows)

    if not merged:
        raise SystemExit("No epoch rows parsed from log.")

    max_epoch = max(int(r.get("epoch", 0)) for r in merged)
    phases = phase_ranges(parsed.args_dict, max_epoch)

    events = validate_rules(merged)
    phase_stat = summarize_phase(merged, phases)
    mech = mechanism_analysis(merged)

    save_csv(merged, out_dir / "merged_epoch.csv")
    save_events_json(out_dir / "validation_report.json", events)
    write_report_md(out_dir / "summary.md", parsed.args_dict, phase_stat, mech, events, merged)

    plot_dashboard(merged, phases, out_dir / "01_overview_phase_dashboard.png")
    plot_loss_ratio(merged, out_dir / "02_loss_contribution_ratio.png")

    # 机器可读总体摘要
    payload = {
        "args": parsed.args_dict,
        "phase_stat": phase_stat,
        "mechanism": mech,
        "event_counts": {
            "ERROR": sum(1 for e in events if e.level == "ERROR"),
            "WARN": sum(1 for e in events if e.level == "WARN"),
            "INFO": sum(1 for e in events if e.level == "INFO"),
        },
        "phases": phases,
        "log": str(log_path),
    }
    (out_dir / "analysis_summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[OK] Analysis saved to: {out_dir}")
    if args.strict_hard and payload["event_counts"]["ERROR"] > 0:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
