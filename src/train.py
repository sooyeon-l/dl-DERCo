import argparse
import json
import random
import torch
import src.config as config
import pandas as pd
import numpy as np
from src.trainer import run_experiment

def set_seed(seed: int):
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(): 

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, required=True)
    parser.add_argument("--model", choices=["cnn", "eegnet"], required=True)
    parser.add_argument("--sfreq", type=int, choices=[250, 1000], default=250)
    parser.add_argument("--window", type=str, required=True)
    parser.add_argument("--shuffle_labels", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    seed = config.RANDOM_SEED
    set_seed(seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"
    
    train_val_path = config.INPUT_PATH
    checkpoint_dir = config.CHECKPOINTS_PATH

    out = config.RUN_OUTPUTS_PATH / args.run
    out.mkdir(parents=True, exist_ok=True)

    results = run_experiment(
        model_name=args.model, 
        window=args.window, 
        train_val_path=train_val_path, 
        device=device, 
        sfreq=args.sfreq, 
        shuffle_labels=args.shuffle_labels, 
        mode='max', 
        run_name=args.run,
        checkpoint_dir=checkpoint_dir,
        output_dir=out,
    )

    pd.DataFrame(results['epoch_log']).to_csv(out / 'epoch_log.csv', index=False)
    pd.DataFrame(results['batch_log']).to_csv(out / 'batch_log.csv', index=False)
    pd.DataFrame(results['best_summary']).to_csv(out / 'best_summary.csv', index=False)
    pd.DataFrame(results['class_balance']).to_csv(out / 'class_balance.csv', index=False)
    np.savez(
        out / "oof_predictions.npz",
        oof_probs=results["oof_probs"],
        oof_labels=results["oof_labels"],
    )
    with open(out / "run_summary.json", "w") as f:
        json.dump(results["run_summary"], f, indent=2)


if __name__ == "__main__":
    main()
