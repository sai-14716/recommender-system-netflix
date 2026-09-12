import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .data import chronological_split, encode_frame, load_ratings, make_id_maps
from .metrics import ranking_metrics, rating_metrics
from .models import BiasModel, ItemResidualKNN, ResidualEnsemble, SVDPlusPlus, TemporalMF, TorchSVDPlusPlus, TorchTemporalMF, save_model


def parse_args():
    parser = argparse.ArgumentParser(description="Run the MovieLens deterministic hybrid experiment.")
    parser.add_argument("--ratings", required=True)
    parser.add_argument("--movies", default=None)
    parser.add_argument("--output", default="artifacts/default")
    parser.add_argument("--max-rows", type=int, default=None)
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--factors", type=int, default=48)
    parser.add_argument("--neighbors", type=int, default=80)
    parser.add_argument("--max-similarity-items", type=int, default=20000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=262144)
    return parser.parse_args()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=float))


def evaluate_model(model, frame):
    predictions = np.clip(model.predict(frame.userIndex.to_numpy(), frame.itemIndex.to_numpy(), frame.timestamp.to_numpy()), 0.5, 5.0)
    return rating_metrics(frame.rating.to_numpy(), predictions)


def evaluate_ranking(model, train, held_out, item_count, limit=1000, k_values=(5, 10, 20)):
    relevant = {}
    seen = {}
    for user, group in train.groupby("userIndex"):
        seen[user] = set(group.itemIndex)
    for user, group in held_out.groupby("userIndex"):
        relevant[user] = set(group.itemIndex)
    recommended = {}
    for user in list(relevant)[:limit]:
        scores = model.score_all(int(user)).copy()
        scores[list(seen.get(user, set()))] = -np.inf
        order = np.argpartition(-scores, kth=max(k_values) - 1)[: max(k_values)]
        recommended[user] = order[np.argsort(-scores[order])].tolist()
    return ranking_metrics(recommended, {user: relevant[user] for user in recommended}, ks=k_values)


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    config = vars(args)
    if config_path.exists() and args.resume:
        config = json.loads(config_path.read_text())
    else:
        write_json(config_path, config)
    started = time.time()
    ratings = load_ratings(config["ratings"], config.get("max_rows"))
    if config.get("max_users"):
        selected = ratings["userId"].drop_duplicates().head(config["max_users"])
        ratings = ratings[ratings.userId.isin(selected)].copy()
    split = chronological_split(ratings)
    user_map, item_map = make_id_maps(split.train)
    train = encode_frame(split.train, user_map, item_map)
    validation = encode_frame(split.validation, user_map, item_map)
    test = encode_frame(split.test, user_map, item_map)
    np.savez_compressed(output / "split.npz", train=train.index.to_numpy(), validation=validation.index.to_numpy(), test=test.index.to_numpy())
    user_count, item_count = len(user_map), len(item_map)
    device = config.get("device", "auto")
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    if device == "cuda":
        base_class, svd_class = TorchTemporalMF, TorchSVDPlusPlus
    else:
        base_class, svd_class = TemporalMF, SVDPlusPlus
    metrics = {}
    bias = BiasModel().fit(train, user_count, item_count)
    metrics["bias_validation"] = evaluate_model(bias, validation)
    save_model(bias, output / "bias.joblib")
    base = base_class(factors=config["factors"], epochs=config["epochs"], seed=42, device=device, batch_size=config.get("batch_size", 262144)).fit(train, user_count, item_count) if device == "cuda" else base_class(factors=config["factors"], epochs=config["epochs"], seed=42).fit(train, user_count, item_count)
    metrics["temporal_mf_validation"] = evaluate_model(base, validation)
    save_model(base, output / "temporal_mf.joblib")
    svdpp = svd_class(factors=config["factors"], epochs=config["epochs"], seed=42, device=device, batch_size=config.get("batch_size", 262144)).fit(train, user_count, item_count) if device == "cuda" else svd_class(factors=config["factors"], epochs=config["epochs"], seed=42).fit(train, user_count, item_count)
    metrics["svdpp_validation"] = evaluate_model(svdpp, validation)
    save_model(svdpp, output / "svdpp.joblib")
    validation_base = base.predict(validation.userIndex, validation.itemIndex, validation.timestamp)
    residual = ItemResidualKNN(neighbors=config["neighbors"], max_items=config["max_similarity_items"]).fit(train, base, item_count)
    validation_residual = residual.predict(validation.userIndex.to_numpy(), validation.itemIndex.to_numpy())
    validation_history = svdpp.predict(validation.userIndex, validation.itemIndex)
    ensemble = ResidualEnsemble(base, residual, history_model=svdpp).fit(
        validation, validation_base, validation_residual, validation_history
    )
    metrics["hybrid_validation"] = evaluate_model(ensemble, validation)
    save_model(residual, output / "residual_knn.joblib")
    save_model(ensemble, output / "hybrid.joblib")
    metrics["hybrid_test"] = evaluate_model(ensemble, test)
    metrics["hybrid_validation_ranking"] = evaluate_ranking(ensemble, train, validation, item_count)
    metrics["hybrid_test_ranking"] = evaluate_ranking(ensemble, train, test, item_count)
    metrics["elapsed_seconds"] = time.time() - started
    write_json(output / "metrics.json", metrics)
    pd.DataFrame([{**{"model": name}, **values} for name, values in metrics.items() if isinstance(values, dict)]).to_csv(output / "results.csv", index=False)
    print(json.dumps(metrics, indent=2, default=float))


if __name__ == "__main__":
    run(parse_args())