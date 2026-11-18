"""MIV regression from financial news DTM and TF-IDF features.

Inputs (produced by 2_sampling_vectorize.py):
    news_dtm_train.csv, news_dtm_test.csv,
    news_tfidf_train.csv, news_tfidf_test.csv

Each CSV requires: symbol, news_vector, MIV.
news_vector is a sparse dictionary encoded as a string, e.g. '{"0": 1.2}'.

Outputs:
    regression_comparison.csv
    top10_company_regression.csv
"""

import ast
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.sparse import csr_matrix
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

SEED = 42
EPOCHS = 10
BATCH_SIZE = 256
LEARNING_RATE = 0.001
HIDDEN_1 = 128
HIDDEN_2 = 64

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def read_dataset(path):
    df = pd.read_csv(path)
    required = {'symbol', 'news_vector', 'MIV'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'{path}: missing columns {sorted(missing)}')
    df['MIV'] = pd.to_numeric(df['MIV'], errors='coerce')
    df['symbol'] = df['symbol'].astype(str).str.strip().str.upper()
    valid = df['MIV'].notna() & np.isfinite(df['MIV'])
    if not valid.all():
        print(f'{path}: dropping {(~valid).sum()} rows with invalid MIV')
        df = df.loc[valid].copy()
    return df.reset_index(drop=True)


def parse_vector(value):
    if isinstance(value, dict):
        obj = value
    elif pd.isna(value) or str(value).strip() == '':
        obj = {}
    else:
        try:
            obj = json.loads(value)
        except (ValueError, TypeError):
            obj = ast.literal_eval(value)
    if not isinstance(obj, dict):
        raise ValueError(f'Expected sparse feature dictionary, got {type(obj).__name__}')
    return {int(k): float(v) for k, v in obj.items() if float(v) != 0.0}


def make_sparse(vectors, n_features=None):
    parsed = [parse_vector(v) for v in vectors]
    if n_features is None:
        n_features = max((max(d) for d in parsed if d), default=-1) + 1
    rows, cols, vals = [], [], []
    for i, d in enumerate(parsed):
        for j, val in d.items():
            if j < 0 or j >= n_features:
                raise ValueError(f'Feature index {j} outside [0, {n_features})')
            rows.append(i)
            cols.append(j)
            vals.append(val)
    matrix = csr_matrix((np.asarray(vals, dtype=np.float32), (rows, cols)),
                        shape=(len(parsed), n_features), dtype=np.float32)
    return matrix


class MLPRegressor(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, HIDDEN_1),
            nn.ReLU(),
            nn.Linear(HIDDEN_1, HIDDEN_2),
            nn.ReLU(),
            nn.Linear(HIDDEN_2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def metrics(y_true, y_pred):
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    # R2 is undefined for a single observation or constant target.
    r2 = float(r2_score(y_true, y_pred)) if len(y_true) >= 2 and np.var(y_true) > 0 else np.nan
    return rmse, r2


def train_and_evaluate(name, train_path, test_path):
    train = read_dataset(train_path)
    test = read_dataset(test_path)
    if train.empty or test.empty:
        raise ValueError(f'{name}: empty training or testing data')

    # Preserve the feature index space from training, and include any valid test indices.
    train_parsed = [parse_vector(v) for v in train['news_vector']]
    test_parsed = [parse_vector(v) for v in test['news_vector']]
    n_features = max((max(d) for d in train_parsed + test_parsed if d), default=-1) + 1
    if n_features == 0:
        raise ValueError(f'{name}: no nonzero text features')
    X_train = make_sparse(train_parsed, n_features).toarray()
    X_test = make_sparse(test_parsed, n_features).toarray()
    y_train = train['MIV'].to_numpy(dtype=np.float64)
    y_test = test['MIV'].to_numpy(dtype=np.float64)

    # MIV may be on a large scale. Standardize targets using training data only.
    scaler = StandardScaler()
    y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).ravel().astype(np.float32)

    model = MLPRegressor(n_features).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train_scaled)),
        batch_size=BATCH_SIZE, shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )

    print(f'\n========== {name} ==========')
    print(f'Training shape: {X_train.shape}')
    print(f'Testing shape: {X_test.shape}')
    print(f'Device: {DEVICE}')
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for features, targets in loader:
            features = features.to(DEVICE)
            targets = targets.to(DEVICE)
            optimizer.zero_grad()
            loss = loss_fn(model(features), targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(targets)
        print(f'Epoch {epoch + 1}/{EPOCHS} Loss: {total_loss / len(train):.6f}')

    model.eval()
    with torch.no_grad():
        scaled_predictions = []
        for start in range(0, len(X_test), BATCH_SIZE):
            features = torch.from_numpy(X_test[start:start + BATCH_SIZE]).to(DEVICE)
            scaled_predictions.append(model(features).cpu().numpy())
    y_pred = scaler.inverse_transform(
        np.concatenate(scaled_predictions).reshape(-1, 1)
    ).ravel()

    rmse, r2 = metrics(y_test, y_pred)
    print(f'{name} Test RMSE: {rmse:.6f}')
    print(f'{name} Test R²: {r2:.6f}')
    result = test[['symbol']].copy()
    if 'date' in test.columns:
        result['date'] = test['date'].values
    result['actual'] = y_test
    result['predicted'] = y_pred
    return {'Vectorization': name, 'RMSE': rmse, 'R2': r2}, result


def main():
    dtm_summary, dtm = train_and_evaluate(
        'DTM', Path('news_dtm_train.csv'), Path('news_dtm_test.csv'))
    tfidf_summary, tfidf = train_and_evaluate(
        'TF-IDF', Path('news_tfidf_train.csv'), Path('news_tfidf_test.csv'))

    # Ensure the two representations are evaluated on exactly the same observations.
    if len(dtm) != len(tfidf) or not dtm['symbol'].equals(tfidf['symbol']):
        raise ValueError('DTM and TF-IDF test datasets are not aligned by row/symbol.')
    if 'date' in dtm and not dtm['date'].astype(str).equals(tfidf['date'].astype(str)):
        raise ValueError('DTM and TF-IDF test datasets have different row dates.')
    if not np.allclose(dtm['actual'], tfidf['actual'], rtol=1e-10, atol=1e-10):
        raise ValueError('DTM and TF-IDF test datasets have different MIV targets.')

    comparison = pd.DataFrame([dtm_summary, tfidf_summary])
    print('\n========== OVERALL COMPARISON ==========')
    print(comparison.to_string(index=False))
    comparison.to_csv('regression_comparison.csv', index=False)

    top_symbols = dtm['symbol'].value_counts().head(10)
    rows = []
    for symbol, count in top_symbols.items():
        mask = dtm['symbol'].eq(symbol)
        dtm_rmse, dtm_r2 = metrics(dtm.loc[mask, 'actual'], dtm.loc[mask, 'predicted'])
        tfidf_rmse, tfidf_r2 = metrics(tfidf.loc[mask, 'actual'], tfidf.loc[mask, 'predicted'])
        rows.append({
            'symbol': symbol,
            'Test_Observations': int(count),
            'DTM_RMSE': dtm_rmse,
            'DTM_R2': dtm_r2,
            'TFIDF_RMSE': tfidf_rmse,
            'TFIDF_R2': tfidf_r2,
        })
    top10 = pd.DataFrame(rows)
    print('\n========== TOP 10 COMPANIES ==========')
    print(top10.to_string(index=False, float_format=lambda x: f'{x:.6f}'))
    top10.to_csv('top10_company_regression.csv', index=False)
    print('\nSaved: regression_comparison.csv, top10_company_regression.csv')


if __name__ == '__main__':
    main()
