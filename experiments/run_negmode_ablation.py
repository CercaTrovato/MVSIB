import argparse
import subprocess
import sys
from pathlib import Path


def run_cmd(cmd, cwd=None):
    print("[Ablation]", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=cwd)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main():
    parser = argparse.ArgumentParser(description="Run neg_mode ablations for MVSIB")
    parser.add_argument("--dataset", default="RGB-D", help="Dataset name, e.g. RGB-D/Cora/Hdigit")
    parser.add_argument("--python", default=sys.executable, help="Python executable to use")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    train_py = repo_root / "train.py"

    common_args = [
        "--dataset", args.dataset,
        "--save_debug_npz", "true",
        "--log_dist_interval", "5",
        "--route_uncertain_only", "true",
        "--alpha_fn", "0.1",
        "--hn_beta", "0.1",
    ]

    run_cmd([args.python, str(train_py), *common_args, "--neg_mode", "batch"], cwd=str(repo_root))
    run_cmd([args.python, str(train_py), *common_args, "--neg_mode", "knn", "--knn_neg_k", "20"], cwd=str(repo_root))
    run_cmd([args.python, str(train_py), *common_args, "--neg_mode", "knn", "--knn_neg_k", "50"], cwd=str(repo_root))


if __name__ == "__main__":
    main()
