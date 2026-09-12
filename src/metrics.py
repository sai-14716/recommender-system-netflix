import numpy as np


def rating_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = np.asarray(predicted, dtype=np.float64) - np.asarray(actual, dtype=np.float64)
    return {"rmse": float(np.sqrt(np.mean(error**2))), "mae": float(np.mean(np.abs(error)))}


def ranking_metrics(recommended: dict[int, list[int]], relevant: dict[int, set[int]], ks=(5, 10, 20)) -> dict[str, float]:
    output = {}
    users = [user for user in relevant if relevant[user]]
    for k in ks:
        recalls, ndcgs, aps = [], [], []
        for user in users:
            hits = [int(item in relevant[user]) for item in recommended.get(user, [])[:k]]
            hit_count = sum(hits)
            recalls.append(hit_count / len(relevant[user]))
            dcg = sum(hit / np.log2(index + 2) for index, hit in enumerate(hits))
            ideal = sum(1 / np.log2(index + 2) for index in range(min(len(relevant[user]), k)))
            ndcgs.append(dcg / ideal if ideal else 0.0)
            running = 0
            precision_sum = 0.0
            for index, hit in enumerate(hits, start=1):
                if hit:
                    running += 1
                    precision_sum += running / index
            aps.append(precision_sum / min(len(relevant[user]), k))
        output[f"recall@{k}"] = float(np.mean(recalls)) if recalls else 0.0
        output[f"ndcg@{k}"] = float(np.mean(ndcgs)) if ndcgs else 0.0
        output[f"map@{k}"] = float(np.mean(aps)) if aps else 0.0
    return output