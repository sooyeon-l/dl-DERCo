from pathlib import Path
import subprocess
import sys

MODELS = ["cnn", "eegnet"]
N_SUBJECTS_LIST = [2, 4, 6, 8, 10, 14, 18]
SEEDS = [1, 2, 3, 4, 5]

WINDOW = "0800"
SFREQ = 250
DEVICE = "cuda"

RUNS_DIR = Path("/workspace/data/runs")

def make_run_name(model: str, n_subjects: int, seed: int) -> str:
    return f"{model}_{WINDOW}_{SFREQ}hz_nsubj{n_subjects:02d}_seed{seed:03d}"

def get_output_dir(model: str, n_subjects: int, seed: int) -> Path:
    return (
        RUNS_DIR
        / "subject_ablation"
        / model
        / f"nsubj{n_subjects:02d}"
        / f"seed{seed:03d}"
    )

def run_is_complete(model: str, n_subjects: int, seed: int) -> bool:
    return (get_output_dir(model, n_subjects, seed) / "run_summary.json").exists()

def launch_run(model: str, n_subjects: int, seed: int):
    run_name = make_run_name(model, n_subjects, seed)

    if run_is_complete(run_name):
        print(f"[SKIP] {run_name} already complete.", flush=True)
        return

    cmd = [
        sys.executable,
        "-m",
        "src.train",
        "--run",
        run_name,
        "--model",
        model,
        "--window",
        WINDOW,
        "--sfreq",
        str(SFREQ),
        "--device",
        DEVICE,
        "--n_subjects",
        str(n_subjects),
        "--ablation_seed",
        str(seed),
    ]

    print("\n" + "=" * 80, flush=True)
    print(f"[START] {run_name}", flush=True)
    print(" ".join(cmd), flush=True)
    print("=" * 80, flush=True)

    subprocess.run(cmd, check=True)

    print(f"[DONE] {run_name}", flush=True)

def main():
    total = len(MODELS) * len(N_SUBJECTS_LIST) * len(SEEDS)
    completed = 0

    print(f"Subject-count ablation grid: {total} total runs", flush=True)

    for model in MODELS:
        for n_subjects in N_SUBJECTS_LIST:
            for seed in SEEDS:
                launch_run(model, n_subjects, seed)
                completed += 1
                print(f"[PROGRESS] {completed}/{total} attempted", flush=True)

    print("\nAll requested subject-count ablation runs are complete.", flush=True)


if __name__ == "__main__":
    main()