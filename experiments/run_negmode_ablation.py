import argparse
import subprocess
import sys


def run_cmd(cmd):
    print("[Ablation]", " ".join(cmd))
    proc = subprocess.run(cmd)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run neg_mode ablations for MVSIB")
    parser.add_argument("--dataset", default="RGB-D", help="Dataset name, e.g. RGB-D/Cora/Hdigit")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use")
    args = parser.parse_args()

    common_args = [
        "--dataset", args.dataset,
        "--save_debug_npz", "true",
        "--log_dist_interval", "5",
        "--route_uncertain_only", "true",
        "--alpha_fn", "0.1",
        "--hn_beta", "0.1",
    ]

    run_cmd([args.python, "train.py", *common_args, "--neg_mode", "batch"])
    run_cmd([args.python, "train.py", *common_args, "--neg_mode", "knn", "--knn_neg_k", "20"])
    run_cmd([args.python, "train.py", *common_args, "--neg_mode", "knn", "--knn_neg_k", "50"])


if __name__ == "__main__":
    main()
