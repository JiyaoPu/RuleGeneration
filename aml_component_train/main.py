import argparse
import json
import os
import sys
import subprocess
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings_json", type=str, required=True)
    parser.add_argument("--outputs_dir", type=str, required=True)
    args = parser.parse_args()

    settings_path = Path(args.settings_json)
    out_dir = Path(args.outputs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(settings_path, "r", encoding="utf-8") as f:
        settings = json.load(f)

    # 1) 生成 settings.txt（审计用）
    settings_txt = "\n".join([f"{k}: {v}" for k, v in settings.items()]) + "\n"
    (out_dir / "settings.txt").write_text(settings_txt, encoding="utf-8")
    (out_dir / "settings.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")

    # 2) 把 settings 映射为 Experiment.py 的 argparse 参数
    # 你原来是 input_0...input_13 等，用 key->flag 的方式拼 args
    exp_args = ["python", "trust_evolution/Experiment.py"]

    # a) 简单策略：把所有 key 变成 --<key> <value>
    # 前提：Experiment.py 支持这些参数名（你现在就是这样）
    for k, v in settings.items():
        exp_args += [f"--{k}", str(v)]

    # b) 强制把输出目录传给 Experiment（你需要在 Experiment.py 里加这个参数并使用它）
    exp_args += ["--output_dir", str(out_dir)]

    print("RUN:", " ".join(exp_args))
    subprocess.run(exp_args, check=True)

if __name__ == "__main__":
    main()