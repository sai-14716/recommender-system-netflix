# MovieLens 25M Deterministic Hybrid Recommender

This project is a classical recommender-systems study on the MovieLens 25M dataset combining temporal matrix factorization, implicit user history factors, and content-based item similarity correction into a single hybrid prediction model.

## What the Project Does

This codebase implements a **four-stage collaborative filtering pipeline**:

### Stage 1: Baseline Bias Model
Computes regularized global, user, and item biases by iterative coordinate descent:

- **Global mean**: μ = mean(all ratings)
- **User bias**: b_u = argmin_b Σ(r_ui - μ - b_i - b_u)² + λ_u · ||b_u||²
- **Item bias**: b_i = argmin_b Σ(r_ui - μ - b_u - b_i)² + λ_i · ||b_i||²

These biases capture systematic user preferences (e.g., "critic" users rate lower) and item appeal (e.g., blockbusters rate higher). Regularization coefficients λ_u and λ_i prevent overfitting.

### Stage 2: Temporal Matrix Factorization (Time-Aware SVD)
Extends standard matrix factorization to capture temporal drift in user and item preferences:

```
r̂_ui(t) = μ + b_u + b_i + (d_u + d_i) · f(t) + <p_u, q_i>
```

Where:
- **p_u** ∈ ℝ^k: static user latent factors (k=48 dimensions)
- **q_i** ∈ ℝ^k: static item latent factors
- **d_u, d_i**: user and item temporal drift coefficients (scalars)
- **f(t)** = (t - t_min) / (t_max - t_min) - 0.5: normalized time (centered, ∈ [-0.5, 0.5])
- **<p_u, q_i>**: dot product of latent factor vectors

Trained via minibatch stochastic gradient descent (SGD) with L2 regularization. The temporal term models how users' tastes evolve and how item popularity changes over time.

### Stage 3: SVD++ with Implicit Rating History
Augments temporal MF with implicit feedback by incorporating each user's historical item interactions:

```
r̂_ui = μ + b_u + b_i + <p_u + (1/√|H_u|) · Σ_{j∈H_u} y_j, q_i>
```

Where:
- **H_u**: set of items rated by user u
- **y_j** ∈ ℝ^k: implicit factor vector for item j (learned jointly)
- **1/√|H_u|**: normalization factor (damping constant prevents extreme influence from prolific users)

The implicit history term **y** captures latent item properties without requiring explicit feature engineering. The model learns to weight historical items based on their factors' alignment with candidate items. This is especially powerful for capturing item affinity patterns (e.g., "users who liked movie A also like movie B").

### Stage 4: Item-Neighborhood Residual Correction (Collaborative Filtering)
Applies **k-nearest neighbors on item residuals** to capture item-item similarity:

1. **Residual computation**: For each (user, item, rating) triple:
   ```
   residual_ui = r_ui - [μ + b_u + b_i + TemporalMF prediction]
   ```
   
2. **Residual matrix construction**: Build a sparse user × item matrix of residuals

3. **Item similarity search**: Compute cosine similarity between items using their residual vectors:
   ```
   similarity(i, j) = <residual_i, residual_j> / (||residual_i|| · ||residual_j||)
   ```
   
4. **k-NN residual prediction**: For target (user, item):
   ```
   residual_pred = Σ_{j ∈ neighbors(item)} similarity(item, j) · residual_u,j / Σ |similarity|
   ```

This stage corrects systematic errors where temporal MF and SVD++ underestimate or overestimate, capturing fine-grained item preferences that latent factor models miss.

### Stage 5: Ridge-Regularized Linear Stacking Ensemble
Combines all three predictions into a single hybrid score:

```
r̂_ui_final = w_0 + w_1 · [bias] + w_2 · [temporal MF] + w_3 · [residual KNN]
```

Weights are learned by **ridge regression** on the validation set:
```
w = argmin_w ||y_val - (intercept || bias || temporal_MF || residual_KNN) · w||² + λ_ridge · ||w||²
```

This deterministic stacking (as opposed to probabilistic ranking losses like BPR) achieves interpretable blending of complementary signals. The ridge penalty prevents overfitting to validation-set quirks.

## Architecture Summary

The hybrid prediction formula is:

```
r̂_ui_hybrid = w_0 
            + w_1 · [Bias(u, i)]
            + w_2 · [Temporal MF(u, i, t)]  
            + w_3 · [Item k-NN Residuals(u, i)]
```

Each component targets a different aspect of user–item interactions:
- **Bias**: captures global, user-level, and item-level trends
- **Temporal MF**: captures evolving latent user preferences and item popularities
- **SVD++**: captures implicit structural patterns in user histories
- **Residual k-NN**: corrects systematic errors with interpretable item-similarity logic

## Full-Dataset Results

The full-data chronological experiment produced these **validation** results:

| Model | RMSE | MAE |
| --- | ---: | ---: |
| Regularized bias | 0.9115 | 0.6972 |
| Time-aware MF | 0.8925 | 0.6688 |
| SVD++ | 0.8756 | 0.6580 |
| **Hybrid** | **0.8519** | **0.6591** |

The hybrid achieved the following **future chronological test** result:

| Metric | Test result |
| --- | ---: |
| RMSE | **0.8740** |
| MAE | **0.6636** |

Compared with pure SVD++, the hybrid reduced validation RMSE from `0.8756` to `0.8519`. This is an improvement of approximately **2.7%**:

```
(0.8756 - 0.8519) / 0.8756 = 2.7%
```

The available results do not support a 27% improvement claim. A 27% figure should not be used unless a separate metric and matched baseline produce that result.

The full run took approximately 66 minutes on Kaggle. The current ranking evaluation reports the hybrid's future-split performance as Recall@20 `0.014` and NDCG@20 `0.0067`; ranking claims require additional experimental validation.

## Implementation Notes

### CPU Training (Default)

The CPU implementation uses:
- **BiasModel**: Alternating least squares (iterative coordinate descent)
- **TemporalMF**: Row-wise stochastic gradient descent with temporal drift
- **SVDPlusPlus**: Row-wise SGD with implicit history factor aggregation
- **ItemResidualKNN**: Sparse cosine similarity search (sklearn NearestNeighbors)

Each model trains independently, serialized per epoch, with ~30 seconds per epoch per model.

### GPU Training (Optional)

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

GPU acceleration (TorchTemporalMF, TorchSVDPlusPlus) uses Adam optimizer on full-batch or minibatch training. The item-neighborhood search remains CPU-backed and sparse, avoiding construction of a dense 62,000-by-62,000 movie similarity matrix.

## Model Hyperparameters

| Parameter | Default | Description |
| --- | ---: | --- |
| `factors` | 48 | Latent factor dimensions (k) |
| `epochs` | 8 | Training epochs (passes over data) |
| `learning_rate` | 0.006 | SGD learning rate (η) |
| `regularization` | 0.04 | L2 penalty on user/item factors (λ) |
| `temporal_regularization` | 0.02 | L2 penalty on drift terms (λ_t) |
| `neighbors` | 80 | k in k-NN item similarity search |
| `shrinkage` | 25.0 | Support shrinkage for KNN similarities |
| `ridge` | 1.0 | Ridge penalty on ensemble weights (λ_ridge) |

## Kaggle Storage and Recovery

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

## Local Setup

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

## Research Safeguards

- All learned statistics use only data before the evaluated interaction (temporal leakage prevention).
- Training-seen movies are removed before ranking evaluation.
- The future test split is reserved for final evaluation after model selection (no hyperparameter tuning on test).
- Results should be reported with support-stratified metrics and multiple seeds or time cutoffs where compute permits.
- MovieLens is an explicit-feedback benchmark, not an unbiased production exposure log; results may not transfer to implicit-feedback settings.

## References

- Koren, Y. (2008). Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model. *KDD*. (SVD++)
- Koren, Y. (2009). Collaborative Filtering with Temporal Dynamics. *KDD*. (Temporal MF)
- Bell, R. M., & Koren, Y. (2007). Lessons from the Netflix Prize Challenge. *ACM SIGKDD Explorations*, 9(2), 75–79. (Baseline + Regularization)
