
# ============================================================
# 3_classify.py
# Information Impact Score (IIS) Classification
# DTM vs TF-IDF using PyTorch MLP
# ============================================================

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

import ast
import numpy as np


# ------------------------------------------------------------
# 1. Settings
# ------------------------------------------------------------

SEED = 42
BATCH_SIZE = 256
EPOCHS = 10
LEARNING_RATE = 0.001

LABEL_COLUMN = "IIS"
SYMBOL_COLUMN = "symbol"

FILES = {
    "DTM": {
        "train": "news_dtm_train.csv",
        "test": "news_dtm_test.csv"
    },
    "TF-IDF": {
        "train": "news_tfidf_train.csv",
        "test": "news_tfidf_test.csv"
    }
}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

print("Device:", device)


# ------------------------------------------------------------
# 2. Define MLP architecture
# ------------------------------------------------------------

class MLP(nn.Module):

    def __init__(self, input_size, num_classes):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.network(x)


# ------------------------------------------------------------
# 3. Load vectorized datasets
# ------------------------------------------------------------
 
def load_data(train_file, test_file):

    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)

    # Parse sparse dictionaries
    train_vectors = train_df["news_vector"].apply(ast.literal_eval)
    test_vectors = test_df["news_vector"].apply(ast.literal_eval)

    # Find the number of features across both datasets
    all_vectors = list(train_vectors) + list(test_vectors)

    max_index = max(
        (int(k) for vec in all_vectors for k in vec.keys()),
        default=-1
    )

    num_features = max_index + 1

    # Convert sparse dictionaries into dense arrays
    def to_dense(vectors):
        X = np.zeros(
            (len(vectors), num_features),
            dtype=np.float32
        )

        for i, vec in enumerate(vectors):
            for feature_id, value in vec.items():
                X[i, int(feature_id)] = float(value)

        return X

    X_train = to_dense(train_vectors)
    X_test = to_dense(test_vectors)

    # IIS classification labels
    encoder = LabelEncoder()

    y_train = encoder.fit_transform(train_df["IIS"])
    y_test = encoder.transform(test_df["IIS"])

    # Company symbols
    test_symbols = test_df["symbol"].to_numpy()

    print("Training shape:", X_train.shape)
    print("Testing shape:", X_test.shape)
    print("IIS classes:", encoder.classes_)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        test_symbols,
        encoder
    )
# ------------------------------------------------------------
# 4. Train MLP
# ------------------------------------------------------------

def train_model(X_train, y_train, num_classes):

    torch.manual_seed(SEED)

    model = MLP(
        input_size=X_train.shape[1],
        num_classes=num_classes
    ).to(device)

    X_tensor = torch.from_numpy(X_train)
    y_tensor = torch.tensor(y_train, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)

    generator = torch.Generator().manual_seed(SEED)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=generator
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    for epoch in range(EPOCHS):

        model.train()
        total_loss = 0

        for X_batch, y_batch in loader:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()

            outputs = model(X_batch)

            loss = criterion(outputs, y_batch)

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item() * len(X_batch)
            )

        average_loss = total_loss / len(dataset)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} "
            f"Loss: {average_loss:.4f}"
        )

    return model


# ------------------------------------------------------------
# 5. Evaluate test accuracy
# ------------------------------------------------------------

def evaluate_model(model, X_test, y_test):

    model.eval()

    predictions = []

    with torch.no_grad():

        for start in range(0, len(X_test), BATCH_SIZE):

            batch = torch.from_numpy(
                X_test[start:start + BATCH_SIZE]
            ).to(device)

            outputs = model(batch)

            predicted = torch.argmax(
                outputs, dim=1
            )

            predictions.extend(
                predicted.cpu().numpy()
            )

    predictions = np.array(predictions)

    accuracy = accuracy_score(
        y_test, predictions
    ) * 100

    return accuracy, predictions


# ------------------------------------------------------------
# 6. Train and compare DTM vs TF-IDF
# ------------------------------------------------------------

results = []
company_results = []

test_symbols_reference = None
test_labels_reference = None

for method, paths in FILES.items():

    print(f"\n========== {method} ==========")

    (
        X_train, X_test,
        y_train, y_test,
        test_symbols, encoder
    ) = load_data(
        paths["train"],
        paths["test"]
    )

    # Get original IIS class labels
    original_test_labels = encoder.inverse_transform(y_test)

    # Save DTM test data, then compare with TF-IDF
    if test_symbols_reference is None:
        test_symbols_reference = test_symbols.copy()
        test_labels_reference = original_test_labels.copy()

    else:
        if not np.array_equal(
            test_symbols_reference, test_symbols
        ):
            raise ValueError(
                "DTM and TF-IDF test symbols do not match."
            )

        if not np.array_equal(
            test_labels_reference, original_test_labels
        ):
            raise ValueError(
                "DTM and TF-IDF test labels do not match."
            )

    # Train the MLP
    model = train_model(
        X_train,
        y_train,
        len(encoder.classes_)
    )

    # Evaluate
    accuracy, predictions = evaluate_model(
        model, X_test, y_test
    )

    print(f"{method} Test Accuracy: {accuracy:.2f}%")

    results.append({
        "Vectorization": method,
        "Test_Accuracy": accuracy
    })

    # Calculate accuracy for each company
    evaluation_df = pd.DataFrame({
        "symbol": test_symbols,
        "correct": predictions == y_test
    })

    company_df = (
        evaluation_df.groupby("symbol")
        .agg(
            Test_Observations=("correct", "size"),
            Accuracy=("correct", "mean")
        )
        .reset_index()
    )

    company_df["Accuracy"] *= 100
    company_df["Vectorization"] = method

    company_results.append(company_df)

# ------------------------------------------------------------
# 7. Compare overall test accuracy
# ------------------------------------------------------------

results_df = pd.DataFrame(results)

print("\n========== OVERALL COMPARISON ==========")
print(results_df.to_string(index=False))

results_df.to_csv(
    "classification_comparison.csv",
    index=False
)


# ------------------------------------------------------------
# 8. Top 10 companies by test observations
# ------------------------------------------------------------

# top_10 = (
#     pd.Series(test_symbols_reference)
#     .value_counts()
#     .head(10)
#     .index
# )

# all_company_results = pd.concat(
#     company_results,
#     ignore_index=True
# )

# top_results = all_company_results[
#     all_company_results["symbol"].isin(top_10)
# ].copy()

# comparison = top_results.pivot(
#     index=["symbol", "Test_Observations"],
#     columns="Vectorization",
#     values="Accuracy"
# ).reset_index()

# comparison = comparison.sort_values(
#     "Test_Observations",
#     ascending=False
# )
# ------------------------------------------------------------
# 8. Top 10 companies by test observations
# ------------------------------------------------------------

# Count test observations for each company
test_counts = pd.Series(test_symbols_reference).value_counts()

# Define top_10 BEFORE using it
top_10 = test_counts.head(10).index.tolist()

print("Top 10 companies:", top_10)

# Combine DTM and TF-IDF company results
all_company_results = pd.concat(
    company_results,
    ignore_index=True
)

# Keep only the top 10 companies
top_results = all_company_results[
    all_company_results["symbol"].isin(top_10)
].copy()

# Compare DTM and TF-IDF accuracy
comparison = top_results.pivot(
    index=["symbol", "Test_Observations"],
    columns="Vectorization",
    values="Accuracy"
).reset_index()

comparison = comparison.sort_values(
    "Test_Observations",
    ascending=False
)

print("\nTop 10 company results:")
print(comparison.to_string(index=False))

comparison.to_csv(
    "top10_company_accuracy.csv",
    index=False
)

print("\n========== TOP 10 COMPANIES ==========")
print(comparison.to_string(index=False))

comparison.to_csv(
    "top10_company_accuracy.csv",
    index=False
)

print("\nSaved:")
print("classification_comparison.csv")
print("top10_company_accuracy.csv")
