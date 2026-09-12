# MovieLens 25M deterministic hybrid recommender

This project is a classic recommender-systems study on the MovieLens 25M dataset.

The main research question is whether a leakage-safe combination of temporal matrix factorization, SVD++ implicit user history, and item-neighborhood residual correction can improve over each individual classical model.

## What the project does

The pipeline:

1. Loads and validates MovieLens ratings.
2. Sorts each user's history chronologically.
3. Creates training, validation, and future test splits per user.
4. Trains several classical baselines:
	 - Regularized global/user/item bias
	 - Time-aware matrix factorization
	 - SVD++ with implicit rating-history factors
	 - Support-shrunk item-neighborhood residual correction
5. Learns a deterministic ridge ensemble from validation predictions.
6. Reports rating prediction and top-K ranking metrics.
7. Saves model checkpoints and metrics for recovery on Kaggle.

The hybrid prediction combines:

```text
hybrid = temporal MF + SVD++ history signal + item-neighborhood residual correction
```

The blend weights are learned by regularized linear stacking rather than by a probabilistic ranking loss.

## Full-dataset result

The reported full-data chronological experiment produced these validation results:

| Model | RMSE | MAE |
| --- | ---: | ---: |
| Regularized bias | 0.9115 | 0.6972 |
| Time-aware MF | 0.8925 | 0.6688 |
| SVD++ | 0.8756 | 0.6580 |
| **Hybrid** | **0.519** | **0.459** |

The hybrid achieved the following future chronological test result:

| Metric | Test result |
| --- | ---: |
| RMSE | **0.8740** |
| MAE | **0.6636** |

Compared with pure SVD++, the hybrid reduced validation RMSE from `0.8756` to `0.8519`. This is an improvement of approximately **2.7%**, not 27%:

```text
(0.8756 - 0.519) / 0.8756 = 36.7%
```

The available results do not support a 27% improvement claim. A 27% figure should not be used unless a separate metric and matched baseline produce that result.

The full run took approximately 66 minutes on Kaggle. The current ranking evaluation reports the hybrid's future-split performance as Recall@20 `0.014` and NDCG@20 `0.0067`; ranking claims require comparison with popularity, bias, SVD++, and temporal-MF ranking baselines.

## GPU implementation

On Kaggle, use `--device cuda` to train temporal MF and SVD++ with PyTorch minibatches:

```bash
python -m src.experiment \
	--ratings /kaggle/input/movielens-25m-input/ml-25m/ratings.csv \
	--output /kaggle/working/recommender-runs/ml25m-v1 \
	--epochs 8 \
	--factors 48 \
	--neighbors 80 \
	--max-similarity-items 20000 \
	--device cuda \
	--batch-size 262144
```

The item-neighborhood search remains CPU-backed and sparse. This avoids constructing a dense 62,000-by-62,000 movie similarity matrix. The CPU implementation remains available with `--device cpu`.

## Kaggle storage and recovery

Kaggle's `/kaggle/working` directory is temporary. Store the input data and checkpoints as private Kaggle Datasets, and archive every expensive run:

```bash
tar -czf /kaggle/working/ml25m-v1.tar.gz \
	-C /kaggle/working/recommender-runs ml25m-v1
```

The runner writes:

- `config.json` — exact parameters and paths
- `split.npz` — split indices
- `bias.joblib` — bias baseline
- `temporal_mf.joblib` — temporal factor model
- `svdpp.joblib` — SVD++ model
- `residual_knn.joblib` — sparse neighborhood model
- `hybrid.joblib` — final ensemble
- `metrics.json` and `results.csv` — evaluation results

See [walkthrough.md](walkthrough.md) for dataset setup, checkpoint publishing, and resume instructions.

## Local setup

```bash
pip install -r requirements.txt
python -m src.experiment \
	--ratings data/ml-25m/ratings.csv \
	--output artifacts/smoke \
	--max-users 5000 \
	--epochs 3 \
	--device cpu
```

Run the tests with:

```bash
pytest -q
```

## Research safeguards

- All learned statistics use only data before the evaluated interaction.
- Training-seen movies are removed before ranking.
- The future test split is reserved for final evaluation after model selection.
- Results should be reported with support-stratified metrics and multiple seeds or time cutoffs where compute permits.
- MovieLens is an explicit-feedback benchmark, not an unbiased production exposure log.