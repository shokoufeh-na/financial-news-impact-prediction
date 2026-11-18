import pandas as pd
import re
from collections import Counter

# Read news data
news = pd.read_csv("news111.csv")


def preprocess_text(text):
    if pd.isna(text):
        return []

    text = text.lower()

    # Remove HTML
    text = re.sub(r"<[^>]+>", " ", text)

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)

    # Remove punctuation and numbers
    text = re.sub(r"[^a-z\s]", " ", text)

    # Tokenize
    tokens = text.split()

    return tokens


# Preprocess headlines
news["tokens"] = news["headline"].apply(preprocess_text)

# Combine all tokens
all_tokens = [
    word
    for tokens in news["tokens"]
    for word in tokens
]

# Common words that are not financially informative
stop_words = {
    "the", "a", "an", "and", "or", "of", "to", "in",
    "for", "on", "at", "with", "as", "by", "from",
    "is", "are", "was", "were", "be", "this", "that",
    "its", "it"
}

filtered_tokens = [
    word
    for word in all_tokens
    if word not in stop_words
    and len(word) > 2
]

# Count word frequencies
word_counts = Counter(filtered_tokens)

# Show most frequent words
print("Most frequent words:")
for word, count in word_counts.most_common(100):
    print(f"{word}: {count}")