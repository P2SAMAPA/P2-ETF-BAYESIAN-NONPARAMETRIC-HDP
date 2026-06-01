import os
import json
from datetime import datetime
import numpy as np
import pandas as pd
from huggingface_hub import HfApi
import config
import data_manager as dm
from hdp_hmm import hdp_hmm_score

def normalize_scores(score_dict):
    # Replace None or NaN with 0
    scores = [v if v is not None and not np.isnan(v) else 0.0 for v in score_dict.values()]
    scores = np.array(scores)
    min_s, max_s = scores.min(), scores.max()
    if max_s - min_s < 1e-12:
        return {ticker: 0.0 for ticker in score_dict}
    norm = (scores - min_s) / (max_s - min_s)
    return {ticker: float(norm[i]) for i, ticker in enumerate(score_dict.keys())}

def run_for_window(returns, window_days):
    if len(returns) < window_days:
        return None
    ret_window = returns.iloc[-window_days:]
    raw_scores = {}
    for ticker in ret_window.columns:
        try:
            s = hdp_hmm_score(ret_window[ticker].values, K=config.TRUNCATION, alpha=config.ALPHA, gamma=config.GAMMA,
                              gibbs_iter=config.GIBBS_ITERATIONS, burn_in=config.BURN_IN)
            if s is None or np.isnan(s):
                s = 0.0
        except Exception as e:
            print(f"    Error for {ticker}: {e}")
            s = 0.0
        raw_scores[ticker] = float(s)
    norm_scores = normalize_scores(raw_scores)
    sorted_norm = sorted(norm_scores.items(), key=lambda x: x[1], reverse=True)
    top_etfs = [{"ticker": t, "hdp_score_norm": s, "raw_score": raw_scores[t]} for t, s in sorted_norm[:config.TOP_N]]
    return {
        "window": window_days,
        "top_etfs": top_etfs,
        "all_scores_raw": raw_scores,
        "all_scores_norm": norm_scores
    }

def main():
    print("Loading master data...")
    dm.load_master_data()
    results = {
        "run_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "windows": config.WINDOWS,
        "truncation": config.TRUNCATION,
        "alpha": config.ALPHA,
        "gamma": config.GAMMA,
        "gibbs_iterations": config.GIBBS_ITERATIONS,
        "burn_in": config.BURN_IN,
        "universes": {}
    }
    for uni_name in config.UNIVERSES.keys():
        print(f"Processing {uni_name}...")
        returns = dm.get_universe_returns(uni_name)
        if returns.empty:
            print("  No data -> skipping")
            continue
        all_window_results = []
        for w in config.WINDOWS:
            print(f"  Window {w} days")
            out = run_for_window(returns, w)
            if out:
                all_window_results.append(out)
            else:
                print(f"    Failed for window {w}")
        best_data = all_window_results[-1] if all_window_results else None
        results["universes"][uni_name] = {
            "best_window_data": best_data,
            "all_windows": all_window_results
        }
    os.makedirs("output", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = f"output/hdp_hmm_{timestamp}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {out_file}")
    api = HfApi(token=config.HF_TOKEN)
    try:
        api.upload_file(
            path_or_fileobj=out_file,
            path_in_repo=os.path.basename(out_file),
            repo_id=config.OUTPUT_REPO,
            repo_type="dataset"
        )
        print(f"Uploaded to {config.OUTPUT_REPO}")
    except Exception as e:
        print(f"Upload failed: {e}")

if __name__ == "__main__":
    main()
