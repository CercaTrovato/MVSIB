#!/usr/bin/env python3
"""
English documentation details.

English documentation details.
English documentation details.
English documentation details.
English documentation details.
English documentation details.
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
    """English documentation for this section.

    level: ERROR/WARN/INFO
    English documentation details.
    English documentation details.
    English documentation details.
    """

    level: str
    rule: str
    epoch: int
    detail: str


@dataclass
class ParsedLog:
    """English documentation for this section.

    English documentation details.
    English documentation details.
    English documentation details.
    """

    args_dict: Dict[str, Any]
    metric_rows: List[Dict[str, Any]]
    route_rows: List[Dict[str, Any]]
    distr_rows: List[Dict[str, Any]]
    final_summary: Dict[str, Any]


def _to_num(v: str) -> Any:
    """English documentation for this section.

    English documentation details.
    English documentation details.
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
    """English documentation for this section.

    English documentation details.
    English documentation details.
    """

    out = {}
    for k, v in KV_RE.findall(seg):
        out[k] = _to_num(v.strip().rstrip(','))
    return out


def parse_args_line(line: str) -> Dict[str, Any]:
    """English documentation for this section.

    English documentation details.
    English documentation details.
    """

    m = re.search(r"Args:\s*Namespace\((.*)\)$", line)
    if not m:
        return {}
    inner = m.group(1)
    # English explanation comment.
    try:
        py_expr = "{" + re.sub(r"(\w+)=", r"'\1':", inner) + "}"
        return ast.literal_eval(py_expr)
    except Exception:
        return {}


def parse_log_file(log_path: Path) -> ParsedLog:
    """English documentation for this section.

    English documentation details.
    English documentation details.
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

    # English explanation comment.
    metric_rows.sort(key=lambda x: x.get("epoch", 0))
    route_rows.sort(key=lambda x: x.get("epoch", 0))
    distr_rows.sort(key=lambda x: x.get("epoch", 0))

    return ParsedLog(args_dict, metric_rows, route_rows, distr_rows, final_summary)


def merge_by_epoch(*rows_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """English documentation for this section.

    English documentation details.
    English documentation details.
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
    """English documentation for this section.

    English documentation details.
    English documentation details.
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
    """Safely parse a float value; return NaN on failure."""

    try:
        return float(v)
    except Exception:
        return float(default)


def validate_rules(rows: List[Dict[str, Any]], eps: float = 5e-3) -> List[RuleEvent]:
    """English documentation for this section.

    English documentation details.
    English documentation details.
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

        # English explanation comment.
        if not (0.0 - eps <= fnr <= 1.0 + eps):
            events.append(RuleEvent("ERROR", "fn_ratio_range", ep, f"FN_ratio={fnr}"))
        if not (0.0 - eps <= hnr <= 1.0 + eps):
            events.append(RuleEvent("ERROR", "hn_ratio_range", ep, f"HN_ratio={hnr}"))
        if not (math.isnan(cand) or math.isnan(filt) or cand + eps >= filt):
            events.append(RuleEvent("ERROR", "candidate_vs_filtered", ep, f"candidate={cand}, filtered={filt}"))
        if not (math.isnan(u_size) or u_size <= 0 or math.isnan(used) or used > 0):
            events.append(RuleEvent("ERROR", "neg_used_nonzero", ep, f"U_size={u_size}, used={used}"))

        # English explanation comment.
        if not (math.isnan(fnc) or math.isnan(neg) or neg <= 0):
            fnr_re = fnc / neg
            if not math.isnan(fnr) and abs(fnr - fnr_re) > 0.05:  # Different counting conventions are tolerated; large deviation raises a warning
                events.append(RuleEvent("WARN", "fn_ratio_recompute_gap", ep, f"FN_ratio={fnr:.4f}, FN_count/neg_count={fnr_re:.4f}"))
        if not (math.isnan(hnc) or math.isnan(safe_neg) or safe_neg <= 0):
            hnr_re = hnc / safe_neg
            if not math.isnan(hnr) and abs(hnr - hnr_re) > 0.05:
                events.append(RuleEvent("WARN", "hn_ratio_recompute_gap", ep, f"HN_ratio={hnr:.4f}, HN_count/safe_neg={hnr_re:.4f}"))

        # English explanation comment.
        if not (math.isnan(fshare) or math.isnan(fnc) or math.isnan(neg) or neg <= 0):
            if abs(fshare - fnc / neg) > 0.02:
                events.append(RuleEvent("ERROR", "fn_pair_share_consistency", ep, f"fn_pair_share={fshare:.4f}, FN_count/neg_count={fnc/neg:.4f}"))
        if not (math.isnan(hshare) or math.isnan(hnc) or math.isnan(safe_neg) or safe_neg <= 0):
            if abs(hshare - hnc / safe_neg) > 0.02:
                events.append(RuleEvent("ERROR", "hn_pair_share_consistency", ep, f"hn_pair_share={hshare:.4f}, HN_count/safe_neg={hnc/safe_neg:.4f}"))

        # English explanation comment.
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
    """English documentation for this section.

    English documentation details.
    English documentation details.
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
    """English documentation for this section.

    English documentation details.
    English documentation details.
    """

    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    return float(np.corrcoef(a[m], b[m])[0, 1])


def mechanism_analysis(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    """English documentation for this section.

    English documentation details.
    English documentation details.
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
    """English documentation for this section.

    English documentation details.
    English documentation details.
    """

    keys = sorted({k for r in rows for k in r.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def plot_dashboard(rows: List[Dict[str, Any]], phases: List[Tuple[str, int, int]], out_png: Path) -> None:
    """English documentation for this section.

    English documentation details.
    English documentation details.
    English documentation details.
    English documentation details.
    English documentation details.
    English documentation details.
    English documentation details.

    English documentation details.
    English documentation details.
    English documentation details.

    English documentation details.
    English documentation details.
    English documentation details.
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
    axes[0, 0].set_title("Performance Curves")
    axes[0, 0].legend()

    axes[0, 1].plot(ep, ltot, label="L_total")
    ax2 = axes[0, 1].twinx()
    ax2.plot(ep, ratio, color="tab:red", label="L_cross/L_total")
    axes[0, 1].set_title("Total Loss and Cross-view Ratio")

    axes[1, 0].plot(ep, fnr, label="FN_ratio")
    axes[1, 0].plot(ep, hnr, label="HN_ratio")
    axes[1, 0].set_title("ROUTE Ratios")
    axes[1, 0].legend()

    axes[1, 1].plot(ep, cand, label="candidate")
    axes[1, 1].plot(ep, flt, label="after_filter")
    axes[1, 1].plot(ep, used, label="used_in_loss")
    axes[1, 1].set_title("Negative Pool Flow")
    axes[1, 1].legend()

    axes[2, 0].plot(ep, up50, label="u_p50")
    axes[2, 0].plot(ep, gp50, label="gamma_p50")
    axes[2, 0].plot(ep, sp50, label="S_p50")
    axes[2, 0].set_title("DISTR Mechanism Signals")
    axes[2, 0].legend()

    axes[2, 1].plot(ep, inc, label="route_inconsistent")
    axes[2, 1].plot(ep, empty, label="empty_cluster")
    axes[2, 1].set_title("Consistency and Cluster Health")
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
    """English documentation for this section.

    English documentation details.
    English documentation details.
    English documentation details.
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
    plt.title("Loss Contribution Ratios")
    plt.xlabel("epoch")
    plt.ylabel("ratio")
    plt.legend()
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=160)
    plt.close()


def write_report_md(path: Path, args_dict: Dict[str, Any], phase_stat: Dict[str, Dict[str, float]],
                    mech: Dict[str, float], events: List[RuleEvent], rows: List[Dict[str, Any]]) -> None:
    """English documentation for this section.

    English documentation details.
    English documentation details.
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

    # English explanation comment.
    bad = [e for e in events if e.level in {"ERROR", "WARN"}][:30]
    if bad:
        lines.append("\n## 主要告警样本（前30条）\n")
        for e in bad:
            lines.append(f"- [{e.level}] epoch={e.epoch} {e.rule}: {e.detail}\n")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines), encoding="utf-8")


def save_events_json(path: Path, events: List[RuleEvent]) -> None:
    """Save rule events as JSON."""

    data = [e.__dict__ for e in events]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    """English documentation for this section.

    English documentation details.
    English documentation details.
    English documentation details.
    English documentation details.
    English documentation details.
    English documentation details.
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

    plot_dashboard(merged, phases, out_dir / "01_overview_phase_dashboard.svg")
    plot_loss_ratio(merged, out_dir / "02_loss_contribution_ratio.svg")

    # English explanation comment.
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
