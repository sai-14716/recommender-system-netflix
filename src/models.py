from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


def _require_torch():
    try:
        import torch
    except ImportError as error:
        raise ImportError("GPU mode requires PyTorch. Kaggle notebooks normally include it; otherwise install a CUDA-compatible torch build.") from error
    return torch


class BiasModel:
    def __init__(self, user_regularization=12.0, item_regularization=8.0):
        self.user_regularization = user_regularization
        self.item_regularization = item_regularization

    def fit(self, frame, user_count, item_count):
        self.global_mean = float(frame.rating.mean())
        self.user_bias = np.zeros(user_count, dtype=np.float32)
        self.item_bias = np.zeros(item_count, dtype=np.float32)
        for _ in range(8):
            residual = frame.rating.to_numpy() - self.global_mean - self.item_bias[frame.itemIndex]
            self.user_bias = np.bincount(frame.userIndex, weights=residual, minlength=user_count) / (self.user_regularization + frame.groupby("userIndex").size().reindex(range(user_count), fill_value=0).to_numpy())
            residual = frame.rating.to_numpy() - self.global_mean - self.user_bias[frame.userIndex]
            self.item_bias = np.bincount(frame.itemIndex, weights=residual, minlength=item_count) / (self.item_regularization + frame.groupby("itemIndex").size().reindex(range(item_count), fill_value=0).to_numpy())
        return self

    def predict(self, users, items, timestamps=None):
        return self.global_mean + self.user_bias[np.asarray(users)] + self.item_bias[np.asarray(items)]

    def score_all(self, user):
        return (self.global_mean + self.user_bias[user] + self.item_bias).astype(np.float32)


class TemporalMF:
    def __init__(self, factors=48, epochs=8, learning_rate=0.006, regularization=0.04, temporal_regularization=0.02, seed=42):
        self.factors, self.epochs, self.learning_rate = factors, epochs, learning_rate
        self.regularization, self.temporal_regularization, self.seed = regularization, temporal_regularization, seed

    def fit(self, frame, user_count, item_count):
        rng = np.random.default_rng(self.seed)
        self.user_factors = (rng.normal(0, 0.05, (user_count, self.factors))).astype(np.float32)
        self.item_factors = (rng.normal(0, 0.05, (item_count, self.factors))).astype(np.float32)
        self.user_bias = np.zeros(user_count, dtype=np.float32)
        self.item_bias = np.zeros(item_count, dtype=np.float32)
        self.user_drift = np.zeros(user_count, dtype=np.float32)
        self.item_drift = np.zeros(item_count, dtype=np.float32)
        self.global_mean = float(frame.rating.mean())
        min_time = frame.timestamp.min()
        scale = max(frame.timestamp.max() - min_time, 1)
        for _ in range(self.epochs):
            for row in frame.itertuples(index=False):
                time_value = (row.timestamp - min_time) / scale - 0.5
                prediction = self.global_mean + self.user_bias[row.userIndex] + self.item_bias[row.itemIndex] + time_value * (self.user_drift[row.userIndex] + self.item_drift[row.itemIndex]) + np.dot(self.user_factors[row.userIndex], self.item_factors[row.itemIndex])
                error = row.rating - prediction
                user_factor = self.user_factors[row.userIndex].copy()
                self.user_bias[row.userIndex] += self.learning_rate * (error - self.regularization * self.user_bias[row.userIndex])
                self.item_bias[row.itemIndex] += self.learning_rate * (error - self.regularization * self.item_bias[row.itemIndex])
                self.user_drift[row.userIndex] += self.learning_rate * (error * time_value - self.temporal_regularization * self.user_drift[row.userIndex])
                self.item_drift[row.itemIndex] += self.learning_rate * (error * time_value - self.temporal_regularization * self.item_drift[row.itemIndex])
                self.user_factors[row.userIndex] += self.learning_rate * (error * self.item_factors[row.itemIndex] - self.regularization * self.user_factors[row.userIndex])
                self.item_factors[row.itemIndex] += self.learning_rate * (error * user_factor - self.regularization * self.item_factors[row.itemIndex])
        self.min_time, self.time_scale = min_time, scale
        return self

    def predict(self, users, items, timestamps=None):
        users, items = np.asarray(users), np.asarray(items)
        if timestamps is None:
            time_value = np.zeros(len(users), dtype=np.float32)
        else:
            time_value = (np.asarray(timestamps) - self.min_time) / self.time_scale - 0.5
        return self.global_mean + self.user_bias[users] + self.item_bias[items] + time_value * (self.user_drift[users] + self.item_drift[items]) + np.sum(self.user_factors[users] * self.item_factors[items], axis=1)

    def score_all(self, user, item_chunk=4096):
        base = self.global_mean + self.user_bias[user] + self.item_bias
        scores = base + self.user_factors[user] @ self.item_factors.T
        return scores.astype(np.float32)


class TorchTemporalMF(TemporalMF):
    """Minibatch temporal MF trained with PyTorch on CPU or CUDA."""

    def __init__(self, *args, device="cuda", batch_size=262144, **kwargs):
        super().__init__(*args, **kwargs)
        self.device_name = device
        self.batch_size = batch_size

    def fit(self, frame, user_count, item_count):
        torch = _require_torch()
        device = torch.device(self.device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false. Enable Kaggle GPU accelerator.")
        generator = torch.Generator(device=device).manual_seed(self.seed)
        users = torch.as_tensor(frame.userIndex.to_numpy(), dtype=torch.long, device=device)
        items = torch.as_tensor(frame.itemIndex.to_numpy(), dtype=torch.long, device=device)
        ratings = torch.as_tensor(frame.rating.to_numpy(), dtype=torch.float32, device=device)
        timestamps = torch.as_tensor(frame.timestamp.to_numpy(), dtype=torch.float32, device=device)
        min_time = float(frame.timestamp.min())
        scale = float(max(frame.timestamp.max() - min_time, 1))
        time_values = (timestamps - min_time) / scale - 0.5
        user_factors = torch.nn.Parameter(torch.randn(user_count, self.factors, device=device, generator=generator) * 0.05)
        item_factors = torch.nn.Parameter(torch.randn(item_count, self.factors, device=device, generator=generator) * 0.05)
        user_bias = torch.nn.Parameter(torch.zeros(user_count, device=device))
        item_bias = torch.nn.Parameter(torch.zeros(item_count, device=device))
        user_drift = torch.nn.Parameter(torch.zeros(user_count, device=device))
        item_drift = torch.nn.Parameter(torch.zeros(item_count, device=device))
        optimizer = torch.optim.Adam([user_factors, item_factors, user_bias, item_bias, user_drift, item_drift], lr=self.learning_rate)
        global_mean = ratings.mean()
        count = len(frame)
        for _ in range(self.epochs):
            order = torch.randperm(count, device=device, generator=generator)
            for start in range(0, count, self.batch_size):
                batch = order[start : start + self.batch_size]
                prediction = global_mean + user_bias[users[batch]] + item_bias[items[batch]]
                prediction = prediction + time_values[batch] * (user_drift[users[batch]] + item_drift[items[batch]])
                prediction = prediction + (user_factors[users[batch]] * item_factors[items[batch]]).sum(dim=1)
                error = prediction - ratings[batch]
                loss = (error * error).mean()
                loss = loss + self.regularization * (user_factors[users[batch]].pow(2).mean() + item_factors[items[batch]].pow(2).mean())
                loss = loss + self.temporal_regularization * (user_drift[users[batch]].pow(2).mean() + item_drift[items[batch]].pow(2).mean())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        self.global_mean = float(global_mean.detach().cpu())
        self.user_factors = user_factors.detach().cpu().numpy().astype(np.float32)
        self.item_factors = item_factors.detach().cpu().numpy().astype(np.float32)
        self.user_bias = user_bias.detach().cpu().numpy().astype(np.float32)
        self.item_bias = item_bias.detach().cpu().numpy().astype(np.float32)
        self.user_drift = user_drift.detach().cpu().numpy().astype(np.float32)
        self.item_drift = item_drift.detach().cpu().numpy().astype(np.float32)
        self.min_time, self.time_scale = min_time, scale
        return self


class SVDPlusPlus:
    def __init__(self, factors=48, epochs=8, learning_rate=0.006, regularization=0.04, seed=42):
        self.factors, self.epochs, self.learning_rate = factors, epochs, learning_rate
        self.regularization, self.seed = regularization, seed

    def fit(self, frame, user_count, item_count):
        rng = np.random.default_rng(self.seed)
        self.user_factors = rng.normal(0, 0.05, (user_count, self.factors)).astype(np.float32)
        self.item_factors = rng.normal(0, 0.05, (item_count, self.factors)).astype(np.float32)
        self.history_factors = np.zeros((item_count, self.factors), dtype=np.float32)
        self.user_bias = np.zeros(user_count, dtype=np.float32)
        self.item_bias = np.zeros(item_count, dtype=np.float32)
        self.global_mean = float(frame.rating.mean())
        histories = frame.groupby("userIndex")["itemIndex"].apply(np.asarray).to_dict()
        for _ in range(self.epochs):
            for row in frame.itertuples(index=False):
                history = histories[row.userIndex]
                normalizer = 1.0 / np.sqrt(max(len(history), 1))
                history_vector = self.history_factors[history].sum(axis=0) * normalizer
                user_vector = self.user_factors[row.userIndex] + history_vector
                prediction = self.global_mean + self.user_bias[row.userIndex] + self.item_bias[row.itemIndex] + np.dot(user_vector, self.item_factors[row.itemIndex])
                error = row.rating - prediction
                item_vector = self.item_factors[row.itemIndex].copy()
                self.user_bias[row.userIndex] += self.learning_rate * (error - self.regularization * self.user_bias[row.userIndex])
                self.item_bias[row.itemIndex] += self.learning_rate * (error - self.regularization * self.item_bias[row.itemIndex])
                self.user_factors[row.userIndex] += self.learning_rate * (error * item_vector - self.regularization * self.user_factors[row.userIndex])
                self.item_factors[row.itemIndex] += self.learning_rate * (error * user_vector - self.regularization * self.item_factors[row.itemIndex])
                self.history_factors[history] += self.learning_rate * (error * item_vector * normalizer - self.regularization * self.history_factors[history])
        self.histories = histories
        return self

    def predict(self, users, items, timestamps=None):
        predictions = np.empty(len(users), dtype=np.float32)
        for index, (user, item) in enumerate(zip(np.asarray(users), np.asarray(items))):
            history = self.histories.get(int(user), np.array([], dtype=np.int32))
            history_vector = self.history_factors[history].sum(axis=0) / np.sqrt(max(len(history), 1))
            user_vector = self.user_factors[user] + history_vector
            predictions[index] = self.global_mean + self.user_bias[user] + self.item_bias[item] + np.dot(user_vector, self.item_factors[item])
        return predictions

    def score_all(self, user):
        history = self.histories.get(int(user), np.array([], dtype=np.int32))
        history_vector = self.history_factors[history].sum(axis=0) / np.sqrt(max(len(history), 1))
        user_vector = self.user_factors[user] + history_vector
        return (self.global_mean + self.user_bias[user] + self.item_bias + self.item_factors @ user_vector).astype(np.float32)


class TorchSVDPlusPlus(SVDPlusPlus):
    """Minibatch SVD++ trained with sparse user-history aggregation on CUDA."""

    def __init__(self, *args, device="cuda", batch_size=262144, **kwargs):
        super().__init__(*args, **kwargs)
        self.device_name = device
        self.batch_size = batch_size

    def fit(self, frame, user_count, item_count):
        torch = _require_torch()
        device = torch.device(self.device_name)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false. Enable Kaggle GPU accelerator.")
        generator = torch.Generator(device=device).manual_seed(self.seed)
        users_np = frame.userIndex.to_numpy()
        items_np = frame.itemIndex.to_numpy()
        users = torch.as_tensor(users_np, dtype=torch.long, device=device)
        items = torch.as_tensor(items_np, dtype=torch.long, device=device)
        ratings = torch.as_tensor(frame.rating.to_numpy(), dtype=torch.float32, device=device)
        counts = np.bincount(users_np, minlength=user_count).astype(np.float32)
        values = 1.0 / np.sqrt(np.maximum(counts[users_np], 1.0))
        history_matrix = torch.sparse_coo_tensor(
            torch.as_tensor(np.vstack([users_np, items_np]), dtype=torch.long, device=device),
            torch.as_tensor(values, dtype=torch.float32, device=device),
            size=(user_count, item_count),
            device=device,
        ).coalesce()
        user_factors = torch.nn.Parameter(torch.randn(user_count, self.factors, device=device, generator=generator) * 0.05)
        item_factors = torch.nn.Parameter(torch.randn(item_count, self.factors, device=device, generator=generator) * 0.05)
        history_factors = torch.nn.Parameter(torch.zeros(item_count, self.factors, device=device))
        user_bias = torch.nn.Parameter(torch.zeros(user_count, device=device))
        item_bias = torch.nn.Parameter(torch.zeros(item_count, device=device))
        optimizer = torch.optim.Adam([user_factors, item_factors, history_factors, user_bias, item_bias], lr=self.learning_rate)
        global_mean = ratings.mean()
        count = len(frame)
        for _ in range(self.epochs):
            order = torch.randperm(count, device=device, generator=generator)
            for start in range(0, count, self.batch_size):
                batch = order[start : start + self.batch_size]
                # Rebuild the graph for each minibatch; autograd graphs cannot be reused after backward().
                user_history = torch.sparse.mm(history_matrix, history_factors)
                effective_user = user_factors[users[batch]] + user_history[users[batch]]
                prediction = global_mean + user_bias[users[batch]] + item_bias[items[batch]]
                prediction = prediction + (effective_user * item_factors[items[batch]]).sum(dim=1)
                error = prediction - ratings[batch]
                loss = (error * error).mean()
                loss = loss + self.regularization * (user_factors[users[batch]].pow(2).mean() + item_factors[items[batch]].pow(2).mean() + history_factors[items[batch]].pow(2).mean())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        self.global_mean = float(global_mean.detach().cpu())
        self.user_factors = user_factors.detach().cpu().numpy().astype(np.float32)
        self.item_factors = item_factors.detach().cpu().numpy().astype(np.float32)
        self.history_factors = history_factors.detach().cpu().numpy().astype(np.float32)
        self.user_bias = user_bias.detach().cpu().numpy().astype(np.float32)
        self.item_bias = item_bias.detach().cpu().numpy().astype(np.float32)
        self.histories = {user: np.flatnonzero(counts[user] > 0) for user in range(user_count)}
        self.histories = {user: frame.loc[frame.userIndex == user, "itemIndex"].to_numpy() for user in range(user_count) if counts[user] > 0}
        return self


class ItemResidualKNN:
    def __init__(self, neighbors=80, shrinkage=25.0, max_items=20000):
        self.neighbors, self.shrinkage, self.max_items = neighbors, shrinkage, max_items

    def fit(self, frame, base_model, item_count):
        residuals = frame.rating.to_numpy() - base_model.predict(frame.userIndex, frame.itemIndex, frame.timestamp)
        rows, cols = frame.userIndex.to_numpy(), frame.itemIndex.to_numpy()
        values = residuals.astype(np.float32)
        matrix = sparse.coo_matrix((values, (rows, cols)), shape=(int(frame.userIndex.max()) + 1, item_count)).tocsr()
        item_support = np.asarray((matrix != 0).sum(axis=0)).ravel()
        self.active_items = np.argsort(-item_support)[: min(self.max_items, item_count)]
        item_matrix = matrix[:, self.active_items].T.tocsr()
        neighbor_count = min(self.neighbors + 1, len(self.active_items))
        search = NearestNeighbors(n_neighbors=neighbor_count, metric="cosine", algorithm="brute", n_jobs=-1)
        search.fit(item_matrix)
        distances, indices = search.kneighbors(item_matrix, return_distance=True)
        self.neighbor_index = self.active_items[indices[:, 1:]]
        self.neighbor_similarity = (1.0 - distances[:, 1:]).astype(np.float32)
        self.active_lookup = {item: index for index, item in enumerate(self.active_items)}
        target_rows = np.repeat(self.active_items, self.neighbor_index.shape[1])
        self.similarity_matrix = sparse.csr_matrix(
            (self.neighbor_similarity.ravel(), (target_rows, self.neighbor_index.ravel())),
            shape=(item_count, item_count),
        )
        self.similarity_denominator = np.asarray(np.abs(self.similarity_matrix).sum(axis=1)).ravel()
        self.user_residuals = matrix
        return self

    def predict(self, users, items):
        predictions = np.zeros(len(users), dtype=np.float32)
        for index, (user, item) in enumerate(zip(users, items)):
            if item not in self.active_lookup:
                continue
            active_index = self.active_lookup[item]
            neighbor_items = self.neighbor_index[active_index]
            weights = self.neighbor_similarity[active_index]
            values = self.user_residuals[user, neighbor_items].toarray().ravel()
            mask = values != 0
            predictions[index] = np.dot(values[mask], weights[mask]) / np.maximum(np.abs(weights[mask]).sum(), 1e-8) if mask.any() else 0.0
        return predictions

    def score_all(self, user):
        numerators = self.user_residuals[user] @ self.similarity_matrix.T
        # SciPy sparse arrays remain sparse when converted with np.asarray().
        # Ranking needs one dense score per item, and the ensemble adds a
        # scalar intercept, which sparse arrays do not support.
        numerators = numerators.toarray().ravel()
        return numerators / np.maximum(self.similarity_denominator, 1e-8)


class ResidualEnsemble:
    def __init__(self, base_model, residual_model, history_model=None, ridge=1.0):
        self.base_model, self.residual_model, self.history_model, self.ridge = base_model, residual_model, history_model, ridge

    def fit(self, validation, base_prediction, residual_prediction, history_prediction=None):
        if self.history_model is not None and history_prediction is None:
            raise ValueError("history_prediction is required when history_model is configured")
        self.history_prediction = history_prediction
        columns = [np.ones(len(validation)), base_prediction]
        if self.history_model is not None:
            columns.append(self.history_prediction)
        columns.append(residual_prediction)
        design = np.column_stack(columns)
        penalty = np.diag([0.0] + [self.ridge] * (design.shape[1] - 1))
        self.weights = np.linalg.solve(design.T @ design + penalty, design.T @ validation.rating.to_numpy())
        return self

    def predict(self, users, items, timestamps=None):
        base = self.base_model.predict(users, items, timestamps)
        residual = self.residual_model.predict(users, items)
        values = [np.ones(len(base)), base]
        if self.history_model is not None:
            values.append(self.history_model.predict(users, items))
        values.append(residual)
        return np.column_stack(values) @ self.weights

    def score_all(self, user):
        base = self.base_model.score_all(user)
        residual = self.residual_model.score_all(user)
        values = [np.ones(len(base)), base]
        if self.history_model is not None:
            values.append(self.history_model.score_all(user))
        values.append(residual)
        return np.column_stack(values) @ self.weights


def save_model(model, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)


def load_model(path: str | Path):
    return joblib.load(path)
