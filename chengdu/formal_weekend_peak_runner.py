from pathlib import Path
import os
import sys

root = Path(__file__).resolve().parent
os.chdir(root)
sys.path.insert(0, str(root))
import Chengdu_mean as runner

runner.RESULTS_DIR = root / "formal_weekend_peak_eta02_3od_3budget_5seed_results"
uncertainty = root.parent / "network" / "Chengdu" / "uncertainty" / "weekend_peak_uncertain_actions.json"
for origin, destination in ((43, 107), (35, 22), (22, 250)):
    args = [
        "--run-all", "--algorithms", "eurac", "pql", "pulse", "segac", "--period", "weekend_peak",
        "--od", str(origin), str(destination),
        "--budget-ratios", "0.975", "1.0", "1.025", "--etas", "0.2",
        "--repeats", "5", "--episodes", "3000", "--mc-runs", "500",
        "--uncertainty-file", str(uncertainty),
    ]
    if runner.main(args):
        raise SystemExit(1)
