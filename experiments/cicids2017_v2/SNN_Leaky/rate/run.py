"""cicids2017_v2 · SNN_Leaky · rate

Thin wrapper. Reads its matching config from configs/ and runs the shared
src.training.run_experiment. No experiment-specific logic lives here.
"""
from pathlib import Path
import sys

HERE = Path(__file__).resolve()
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT))

import yaml
from src.training import run_experiment


def _config_path() -> Path:
    parts = list(HERE.relative_to(ROOT).parts)
    parts[0] = "configs"
    parts[-1] = "config.yaml"
    return ROOT.joinpath(*parts)


def main():
    cfg = yaml.safe_load(open(_config_path()))
    for key in ("results_dir", "summary_csv"):
        if key in cfg and not Path(cfg[key]).is_absolute():
            cfg[key] = str(ROOT / cfg[key])
    record = run_experiment(cfg)
    m = record["test_plus"]["metrics"]
    b = record["test_plus"]["binary_ids_metrics"]
    print(f"[done] {cfg['dataset']}/SNN_{cfg['neuron']}/{cfg['encoding']}"
          f" acc={m['accuracy']:.4f} f1m={m['f1_macro']:.4f}"
          f" MCC={m['matthews_corrcoef']:.4f}"
          f" DR={b['detection_rate']:.4f} FAR={b['false_alarm_rate']:.4f}")


if __name__ == "__main__":
    main()
