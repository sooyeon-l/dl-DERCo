from pathlib import Path
import subprocess
import sys


MODELS = ["cnn_v2"]
N_SUBJECTS_LIST = [2, 4, 6, 8, 10, 12, 14, 18]
SEEDS = [1, 2, 3, 4, 5]

WINDOW = "0800"
SFREQ = 250
DEVICE = "cuda"

RUNS_DIR = Path("/workspace/data/DERCo/outputs/runs")


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


def seeds_for_subject_count(n_subjects: int) -> list[int]:
    # When n_subjects == 18, all seeds select the same full train/val subject pool.
    # So run it once only.
    if n_subjects == 18:
        return [1]

    return SEEDS


def launch_run(model: str, n_subjects: int, seed: int):
    run_name = make_run_name(model, n_subjects, seed)

    if run_is_complete(model, n_subjects, seed):
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
    planned_runs = [
        (model, n_subjects, seed)
        for model in MODELS
        for n_subjects in N_SUBJECTS_LIST
        for seed in seeds_for_subject_count(n_subjects)
    ]

    total = len(planned_runs)

    print(f"Subject-count ablation grid: {total} planned runs", flush=True)
    print(f"Models: {MODELS}", flush=True)
    print(f"Subject counts: {N_SUBJECTS_LIST}", flush=True)
    print("Seeds: 1–5 except n_subjects=18 uses seed 1 only", flush=True)

    for i, (model, n_subjects, seed) in enumerate(planned_runs, start=1):
        launch_run(model, n_subjects, seed)
        print(f"[PROGRESS] {i}/{total} attempted", flush=True)

    print("\nAll requested subject-count ablation runs are complete.", flush=True)


if __name__ == "__main__":
    main()