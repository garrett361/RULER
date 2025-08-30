import argparse
from pathlib import Path

import pandas as pd
import wandb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seqlens",
        type=str,
        help="Comma-separated list of seq lens to report scores for",
    )
    parser.add_argument(
        "--root_dir", type=str, help="Base dir (parent of 'synthetic' dir)"
    )
    parser.add_argument("--tracker_dir", type=str)
    parser.add_argument("--project_name", type=str)
    parser.add_argument("--run_id", type=str)
    parser.add_argument("--num_samples", type=int)
    parser.add_argument("--model_template_type", type=str)
    parser.add_argument("--model_framework", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--batch_size", type=str)

    args = parser.parse_args()
    cfg = {
        "num_samples": args.num_samples,
        "model_template_type": args.model_template_type,
        "model_framework": args.model_framework,
        "model_path": args.model_path,
        "root_dir": args.root_dir,
    }

    wandb.init(
        project=args.project_name,
        dir=args.tracker_dir,
        resume="allow",
        id=args.run_id,
        config=cfg,
        settings=wandb.Settings(init_timeout=3600),
    )
    for seqlen in args.seqlens.split(","):
        csv_path = Path(args.root_dir) / "synthetic" / seqlen / "pred" / "summary.csv"

        df = pd.read_csv(csv_path)
        df = df.set_index(df.columns[0])
        scores = df.loc["Score"].astype(float)
        tasks = df.loc["Tasks"]
        mean_score = scores.mean().item()
        vals_to_track = {
            "seqlen": int(seqlen),
            "mean_score": mean_score,
            "batch_size": args.batch_size,
        }
        vals_to_track = {**vals_to_track, **dict(zip(tasks, scores))}

        # Don't provide a step; wandb will increment the step when resuming a run and the user can
        # choose seqlen for the x-axis in plots later.
        wandb.log(vals_to_track)
