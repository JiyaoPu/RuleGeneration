import argparse
import json
import os
import subprocess
from pathlib import Path


def pick_writable_output_dir(cli_path: str) -> Path:
    """
    In some AML job modes, the provided output mount path might not exist or be writable.
    We fall back to standard AML env vars or local ./outputs.
    """
    candidates = []
    if cli_path:
        candidates.append(Path(cli_path))

    for env_key in ["AZUREML_OUTPUT_DIR", "AZUREML_JOB_OUTPUT_PATH", "AZUREML_ARTIFACTS_DEFAULT_DIR"]:
        v = os.environ.get(env_key)
        if v:
            candidates.append(Path(v))

    candidates.append(Path.cwd() / "outputs")

    for p in candidates:
        try:
            p = p.resolve()
        except Exception:
            pass
        try:
            p.mkdir(parents=True, exist_ok=True)
            probe = p / ".write_probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return p
        except Exception:
            continue

    raise PermissionError("No writable output directory found. Try Output mode = Upload.")


def as_int(x, default=0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def as_float(x, default=0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def as_bool_str(x, default="False") -> str:
    """
    Match Experiment.py's str2bool usage. Return 'True' or 'False' strings.
    """
    if isinstance(x, bool):
        return "True" if x else "False"
    s = str(x).strip().lower()
    if s in ["true", "1", "yes", "y"]:
        return "True"
    if s in ["false", "0", "no", "n"]:
        return "False"
    return default


def ensure_azure_packages():
    """
    Install azure packages only if missing.
    """
    try:
        import azure.identity  # noqa: F401
        import azure.storage.blob  # noqa: F401
        return
    except Exception:
        print("[main] azure packages not found, installing azure-identity & azure-storage-blob ...", flush=True)
        subprocess.run(
            ["python", "-m", "pip", "install", "-q", "azure-identity", "azure-storage-blob"],
            check=True
        )


def upload_outputs_to_blob(local_dir: Path, account_url: str, container: str, prefix: str):
    """
    Upload all files under local_dir to Blob Storage: container/prefix/<relative_path>
    Uses Managed Identity via DefaultAzureCredential.
    """
    ensure_azure_packages()
    from azure.identity import DefaultAzureCredential
    from azure.storage.blob import BlobServiceClient

    cred = DefaultAzureCredential()
    bsc = BlobServiceClient(account_url=account_url, credential=cred)
    cc = bsc.get_container_client(container)

    uploaded = 0
    for p in local_dir.rglob("*"):
        if not p.is_file():
            continue
        rel = p.relative_to(local_dir).as_posix()
        blob_name = f"{prefix.rstrip('/')}/{rel}"
        with p.open("rb") as f:
            cc.upload_blob(name=blob_name, data=f, overwrite=True)
        uploaded += 1

    print(f"[upload] uploaded_files={uploaded} -> {account_url}/{container}/{prefix}/", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings_json", type=str, required=True)
    parser.add_argument("--outputs_dir", type=str, required=True)
    args = parser.parse_args()

    settings_path = Path(args.settings_json)
    if settings_path.is_dir():
        candidate = settings_path / "settings.json"
        if candidate.exists():
            settings_path = candidate
        else:
            raise IsADirectoryError(f"settings_json points to a directory: {settings_path}. Expected a file.")
    if not settings_path.exists():
        raise FileNotFoundError(f"settings_json not found: {settings_path}")

    out_dir = pick_writable_output_dir(args.outputs_dir)
    print(f"[main] settings_json = {settings_path}", flush=True)
    print(f"[main] outputs_dir   = {out_dir}", flush=True)

    with open(settings_path, "r", encoding="utf-8") as f:
        settings = json.load(f)

    # ---- audit: save settings into outputs ----
    settings_txt = "\n".join([f"{k}: {v}" for k, v in settings.items()]) + "\n"
    (out_dir / "settings.txt").write_text(settings_txt, encoding="utf-8")
    (out_dir / "settings.json").write_text(
        json.dumps(settings, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # ---- Map frontend keys -> Experiment.py args ----
    exp_args = ["python", "trust_evolution/Experiment.py"]

    # Rule: population counts (input_0..7)
    exp_args += ["--random_count", str(as_int(settings.get("input_0", 4), 4))]
    exp_args += ["--cheater_count", str(as_int(settings.get("input_1", 4), 4))]
    exp_args += ["--cooperator_count", str(as_int(settings.get("input_2", 4), 4))]
    exp_args += ["--copycat_count", str(as_int(settings.get("input_3", 4), 4))]
    exp_args += ["--grudger_count", str(as_int(settings.get("input_4", 4), 4))]
    exp_args += ["--detective_count", str(as_int(settings.get("input_5", 4), 4))]
    exp_args += ["--ai_count", str(as_int(settings.get("input_6", 1), 1))]
    exp_args += ["--human_count", str(as_int(settings.get("input_7", 2), 2))]

    # Rule: trade_rules (input_8..13) -> 6 floats
    trade_keys = ["input_8", "input_9", "input_10", "input_11", "input_12", "input_13"]
    trade_defaults = [0, 0, 3, -1, 2, 2]
    trade_vals = []
    for i, k in enumerate(trade_keys):
        trade_vals.append(as_float(settings.get(k, trade_defaults[i]), trade_defaults[i]))
    exp_args += ["--trade_rules"] + [str(v) for v in trade_vals]

    # Rule: round/reproduction/mistake
    exp_args += ["--round_number", str(as_int(settings.get("roundNumberInput", 3), 3))]
    exp_args += ["--reproduction_number", str(as_int(settings.get("reproductionNumberInput", 0), 0))]
    exp_args += ["--mistake_possibility", str(as_float(settings.get("mistakePossibilityInput", 0.0), 0.0))]

    # Rule: fixed_rule
    exp_args += ["--fixed_rule", as_bool_str(settings.get("hiddenFixedRule", "True"), "True")]

    # Rule for Game Flow: extrinsic_reward (2 floats)
    if "extrinsic_reward" in settings:
        try:
            er = settings["extrinsic_reward"]
            if isinstance(er, (list, tuple)) and len(er) == 2:
                exp_args += ["--extrinsic_reward", str(as_float(er[0], 0.0)), str(as_float(er[1], 0.0))]
        except Exception:
            pass
    else:
        if ("extrinsic_reward_A" in settings) or ("extrinsic_reward_B" in settings):
            exp_args += [
                "--extrinsic_reward",
                str(as_float(settings.get("extrinsic_reward_A", 0.0), 0.0)),
                str(as_float(settings.get("extrinsic_reward_B", 0.0), 0.0)),
            ]

    # Strategy
    exp_args += ["--humanPlayer", as_bool_str(settings.get("hiddenHumanPlayer", "False"), "False")]
    exp_args += ["--ai_type", str(settings.get("hiddenAIType", "Q"))]

    # Evaluation
    exp_args += ["--cooperationRate", str(as_float(settings.get("cooperationRateInput", 1), 1.0))]
    exp_args += ["--individualIncome", str(as_float(settings.get("individualIncomeInput", 2), 2.0))]
    exp_args += ["--giniCoefficient", str(as_float(settings.get("giniCoefficientInput", 0.50), 0.50))]

    # Evaluation for Game Flow
    if "difficulty" in settings:
        exp_args += ["--difficulty", str(as_float(settings.get("difficulty", 0.01), 0.01))]

    # Designer and Evaluator
    exp_args += ["--batch_size", str(as_int(settings.get("de_input_batch_size", 1), 1))]
    exp_args += ["--lr", str(as_float(settings.get("de_input_lr", 0.01), 0.01))]
    exp_args += ["--b1", str(as_float(settings.get("de_input_b1", 0.5), 0.5))]
    exp_args += ["--b2", str(as_float(settings.get("de_input_b2", 0.999), 0.999))]
    exp_args += ["--RuleDimension", str(as_int(settings.get("de_input_RuleDimension", 3), 3))]
    exp_args += ["--DE_train_episode", str(as_int(settings.get("de_input_DE_train_episode", 1), 1))]
    exp_args += ["--DE_test_episode", str(as_int(settings.get("de_input_DE_test_episode", 1), 1))]
    exp_args += ["--layersNum", str(as_int(settings.get("de_input_layersNum", 1), 1))]
    exp_args += ["--evaluationSize", str(as_int(settings.get("de_input_evaluationSize", 1), 1))]

    # Agent training
    exp_args += ["--agent_train_epoch", str(as_int(settings.get("input_agent_train_epoch", 10), 10))]
    exp_args += ["--gamma", str(as_float(settings.get("input_gamma", 0.99), 0.99))]
    exp_args += ["--epsilon", str(as_float(settings.get("input_epsilon", 1.0), 1.0))]
    exp_args += ["--epsilon_decay", str(as_float(settings.get("input_epsilon_decay", 0.999), 0.999))]
    exp_args += ["--epsilon_min", str(as_float(settings.get("input_epsilon_min", 0.1), 0.1))]
    exp_args += ["--memory_size", str(as_int(settings.get("input_memory_size", 10000), 10000))]
    exp_args += ["--target_update", str(as_int(settings.get("input_target_update", 10), 10))]
    exp_args += ["--state_size", str(as_int(settings.get("input_state_size", 20), 20))]

    # Pass output_dir into Experiment.py (you already added this arg in Experiment.py)
    exp_args += ["--output_dir", str(out_dir)]

    print("[main] RUN:", " ".join(exp_args), flush=True)
    subprocess.run(exp_args, check=True)
    print("[main] Experiment finished.", flush=True)

    # ---- Sync outputs to your Blob container/path (scheme B) ----
    # Use run id to avoid overwriting previous runs
    run_id = os.getenv("AZUREML_RUN_ID") or os.getenv("AZUREML_JOB_ID") or "local"
    prefix = f"run/{run_id}"

    upload_outputs_to_blob(
        local_dir=out_dir,
        account_url="https://rgnspace3954763138.blob.core.windows.net",
        container="rgnresults",
        prefix=prefix,
    )

    print(f"[main] DONE. You can access: rgnresults/{prefix}/...", flush=True)


if __name__ == "__main__":
    main()