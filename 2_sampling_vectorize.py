"""
2_sampling_vectorize.py

Purpose
-------
1. Load news.csv and historical_impacts_values.csv.
2. Convert each news publication timestamp to an event trading date.
3. Match news events with IIS and MIV.
4. Randomly select a consecutive 7-year period using np.random.seed(42).
5. Split the selected period chronologically into 80% training / 20% testing.
6. Prevent target leakage from the 3-trading-day CAR window.
7. Fit DTM and TF-IDF vectorizers on TRAINING NEWS ONLY.
8. Transform both training and testing news using the fitted vectorizers.
9. Save four output datasets.

Outputs
-------
news_dtm_train.csv
news_dtm_test.csv
news_tfidf_train.csv
news_tfidf_test.csv
"""

from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import (
    CountVectorizer,
    TfidfVectorizer
)


# ============================================================
# CONSTANTS
# ============================================================

NEWS_FILE = Path("news111.csv")

IMPACT_FILE = Path(
    "historical_impacts_values-latest.csv"
)

MARKET_SYMBOL = "^sp500"

POSSIBLE_START_YEARS = [
    2009,
    2010,
    2011,
    2012,
    2013,
    2014,
]

TRAIN_RATIO = 0.80

EVENT_WINDOW_DAYS = 3

RANDOM_SEED = 42


# ============================================================
# EVENT DATE
# ============================================================

def assign_event_date(
    news_timestamp,
    trading_dates
):
    """
    Convert a news publication timestamp to its corresponding
    event trading date.

    Rules
    -----
    1. Before 4:00 PM ET on a trading day:
       same trading day.

    2. At or after 4:00 PM ET:
       next trading day.

    3. Weekend / market holiday:
       next trading day.
    """

    if pd.isna(news_timestamp):
        return pd.NaT

    timestamp = pd.Timestamp(news_timestamp)

    # # --------------------------------------------------------
    # # Convert timezone-aware timestamps to U.S. Eastern.
    # # If timestamp is timezone-naive, assume Eastern.
    # # --------------------------------------------------------

    # if timestamp.tzinfo is not None:

    #     timestamp = timestamp.tz_convert(
    #         "US/Eastern"
    #     )

    # else:

    #     timestamp = timestamp.tz_localize(
    #         "US/Eastern"
    #     )

    # publication_date = (
    #     timestamp
    #     .tz_localize(None)
    #     .normalize()
    # )

    
    # --------------------------------------------------------
    # Timestamp is already in Eastern Time.
    # No timezone conversion is needed.
    # --------------------------------------------------------

    timestamp = pd.Timestamp(news_timestamp)

    # Extract calendar date without timezone
    publication_date = pd.Timestamp(timestamp.date())

   # Extract publication time
    publication_time = timestamp.time()


    market_close = pd.Timestamp(
        "16:00"
    ).time()

    # --------------------------------------------------------
    # News at/after 4 PM belongs to next trading day.
    # --------------------------------------------------------

    if publication_time >= market_close:

        candidates = trading_dates[
            trading_dates > publication_date
        ]

    else:

        candidates = trading_dates[
            trading_dates >= publication_date
        ]

    if len(candidates) == 0:
        return pd.NaT

    return pd.Timestamp(
        candidates[0]
    )


# ============================================================
# VECTOR SERIALIZATION
# ============================================================

def sparse_row_to_string(row):
    """
    Store one sparse news vector as a compact JSON dictionary.

    Example:
        {"12": 1, "83": 2, "100": 1}

    Only non-zero features are stored.
    """

    return json.dumps({
        str(int(index)): float(value)
        for index, value
        in zip(row.indices, row.data)
    })


def make_output(
    data,
    matrix,
    date_column="event_date"
):
    """
    Create the assignment output format:

        date
        symbol
        news_vector
        IIS
        MIV
    """

    output = data[
        [
            date_column,
            "symbol",
            "IIS",
            "MIV",
        ]
    ].copy()

    output = output.rename(
        columns={
            date_column: "date"
        }
    )

    output["news_vector"] = [
        sparse_row_to_string(
            matrix.getrow(i)
        )
        for i in range(
            matrix.shape[0]
        )
    ]

    return output[
        [
            "date",
            "symbol",
            "news_vector",
            "IIS",
            "MIV",
        ]
    ]


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # 1. LOAD INPUT FILES
    # ========================================================

    if not NEWS_FILE.exists():

        raise FileNotFoundError(
            f"Cannot find {NEWS_FILE.resolve()}"
        )

    if not IMPACT_FILE.exists():

        raise FileNotFoundError(
            f"Cannot find {IMPACT_FILE.resolve()}"
        )

    news = pd.read_csv(
        NEWS_FILE
    )

    impacts = pd.read_csv(
        IMPACT_FILE
    )


    # ========================================================
    # 2. CHECK REQUIRED COLUMNS
    # ========================================================

    required_news_columns = {
        "date",
        "symbol",
    }

    missing_news = (
        required_news_columns
        - set(news.columns)
    )

    if missing_news:

        raise ValueError(
            "news.csv is missing columns: "
            f"{sorted(missing_news)}"
        )


    required_impact_columns = {
        "date",
        "symbol",
        "IIS",
        "MIV",
    }

    missing_impacts = (
        required_impact_columns
        - set(impacts.columns)
    )

    if missing_impacts:

        raise ValueError(
            "historical_impacts_values.csv "
            "is missing columns: "
            f"{sorted(missing_impacts)}"
        )

    # ============================================================
    # 3. EXTRACT HEADLINE + MATCHING TIMESTAMP
    # ============================================================
    #
    # news.csv contains one row per company + calendar day.
    #
    # If multiple articles exist in that row:
    #
    # date:
    #   timestamp1 || timestamp2 || timestamp3
    #
    # headline:
    #   headline1 || headline2 || headline3
    #
    # The positions correspond:
    #
    #   timestamp1 <-> headline1
    #   timestamp2 <-> headline2
    #   timestamp3 <-> headline3
    #
    # Split them together so every headline keeps its own
    # publication timestamp.
    # ============================================================

    news["timestamp"] = (
        news["date"]
        .fillna("")
        .astype(str)
        .str.split("\x1f", regex=False)
    )

    news["headline"] = (
        news["headline"]
        .fillna("")
        .astype(str)
        .str.split("\x1f", regex=False)
    )


    # Make sure timestamp/headline counts match.
    timestamp_count = news["timestamp"].str.len()
    headline_count = news["headline"].str.len()

    mismatch = timestamp_count != headline_count

    if mismatch.any():

        errors = news.loc[mismatch].copy()

        errors["timestamp_count"] = timestamp_count[mismatch]
        errors["headline_count"] = headline_count[mismatch]

        print(
            errors[
                ["symbol", "timestamp_count", "headline_count"]
            ].head(20).to_string(index=False)
        )

        errors.to_csv("mismatched_news.csv", index=False)

        raise ValueError(
            f"{mismatch.sum():,} rows have different numbers "
            "of timestamps and headlines."
        )


    # Convert each company-day row back into individual articles.
    news = news.explode(
        ["timestamp", "headline"],
        ignore_index=True
    )


    # Clean text.
    news["timestamp"] = (
        news["timestamp"]
        .astype(str)
        .str.strip()
    )

    news["headline"] = (
        news["headline"]
        .astype(str)
        .str.strip()
    )


    # Remove empty articles.
    news = news.loc[
        (news["timestamp"] != "")
        &
        (news["headline"] != "")
    ].copy()


    # ========================================================
    # 4. CLEAN IMPACT DATA
    # ========================================================

    impacts["date"] = pd.to_datetime(
        impacts["date"],
        errors="coerce"
    ).dt.normalize()

    impacts["symbol"] = (
        impacts["symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    impacts["IIS"] = pd.to_numeric(
        impacts["IIS"],
        errors="coerce"
    )

    impacts["MIV"] = pd.to_numeric(
        impacts["MIV"],
        errors="coerce"
    )


    # ========================================================
    # 5. GET ACTUAL TRADING CALENDAR
    # ========================================================
    #
    # Use ^sp500 dates from historical_impacts_values.csv.
    # Weekends and market holidays are therefore absent.
    # ========================================================

    market_rows = impacts.loc[
        impacts["symbol"].str.upper()
        == MARKET_SYMBOL.upper()
    ].copy()

    if market_rows.empty:

        raise ValueError(
            f"No {MARKET_SYMBOL} rows found "
            "in historical_impacts_values.csv."
        )

    trading_dates = (
        pd.DatetimeIndex(
            market_rows["date"]
        )
        .dropna()
        .normalize()
        .sort_values()
        .unique()
    )

    # ============================================================
    # 6. PARSE NEWS TIMESTAMP
    # ============================================================
    #
    # Timestamps in news.csv already contain Eastern offsets:
    #
    #   -05:00 = Eastern Standard Time
    #   -04:00 = Eastern Daylight Time
    #
    # Therefore, preserve the timestamp as stored.
    # ============================================================

    news["timestamp"] = news["timestamp"].apply(
        lambda x: pd.Timestamp(x)
    )

    # ============================================================
    # 7. CREATE EVENT DATE FOR EACH HEADLINE
    # ============================================================

    news["event_date"] = news["timestamp"].apply(
        lambda timestamp: assign_event_date(
            timestamp,
            trading_dates
        )
    )


    news = news.dropna(
        subset=[
            "event_date",
            "symbol",
        ]
    ).copy()


    news["symbol"] = (
        news["symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
    )


    # ============================================================
    # 7B. COMBINE HEADLINES THAT HAVE THE SAME EVENT DATE
    # ============================================================
    #
    # Different publication timestamps may map to the same
    # trading event date.
    #
    # Example:
    #
    #   08:00 -> June 11
    #   10:00 -> June 11
    #   17:00 -> June 12
    #
    # Therefore:
    #
    # June 11 document = headline1 || headline2
    # June 12 document = headline3
    #
    # This produces one news document per:
    #
    #       event_date + symbol
    #
    # ============================================================

    news = (
        news
        .sort_values(
            ["event_date", "symbol", "timestamp"]
        )
        .groupby(
            ["event_date", "symbol"],
            as_index=False
        )
        .agg({
            "headline": lambda x:
                " || ".join(x.astype(str))
        })
    )


    # The headline is the document used for vectorization.

    news["news_text"] = news["headline"]



    print("\nBefore matching:")
    print("News companies:", news["symbol"].nunique())
    print("News observations:", len(news))

    print("\nHistorical impact companies:")
    print(impacts["symbol"].value_counts().head(20))

    print("\nCompanies with valid IIS and MIV:")
    valid_impacts = impacts.dropna(subset=["IIS", "MIV"])
    print(valid_impacts["symbol"].value_counts().head(20))
    # ========================================================
    # 8. MATCH NEWS WITH IIS AND MIV
    # ========================================================
    #
    # news.event_date
    #      +
    # news.symbol
    #
    # matches:
    #
    # impacts.date
    #      +
    # impacts.symbol
    # ========================================================

    financial_targets = impacts.loc[
        impacts["symbol"].str.upper()
        != MARKET_SYMBOL.upper(),
        [
            "date",
            "symbol",
            "IIS",
            "MIV",
        ]
    ].copy()

    financial_targets = (
        financial_targets
        .drop_duplicates(
            ["date", "symbol"]
        )
    )

    matched = news.merge(
        financial_targets,
        left_on=[
            "event_date",
            "symbol",
        ],
        right_on=[
            "date",
            "symbol",
        ],
        how="inner",
        suffixes=(
            "_news",
            "_impact",
        )
    )

    # Keep only rows with valid targets.

    matched = matched.dropna(
        subset=[
            "IIS",
            "MIV",
        ]
    ).copy()


    print("\nAfter matching:")
    print("Matched companies:", matched["symbol"].nunique())
    print("Matched observations:", len(matched))
    print(matched["symbol"].value_counts().head(20))

    print("\n=== MATCHED OBSERVATIONS ===")
    print("Companies:", matched["symbol"].nunique())
    print(matched["symbol"].value_counts())

    print("\n=== MATCHED DATE RANGES ===")
    print(
        matched.groupby("symbol")["event_date"]
        .agg(["min", "max", "count"])
    )

    # ========================================================
    # 9. RANDOMLY SELECT CONSECUTIVE 7-YEAR PERIOD
    # ========================================================

    np.random.seed(
        RANDOM_SEED
    )

    start_year = int(
        np.random.choice(
            POSSIBLE_START_YEARS
        )
    )

    end_year = (
        start_year + 6
    )

    period_start = pd.Timestamp(
        year=start_year,
        month=1,
        day=1
    )

    period_end = pd.Timestamp(
        year=end_year,
        month=12,
        day=31
    )

    selected = matched.loc[
        (
            matched["event_date"]
            >= period_start
        )
        &
        (
            matched["event_date"]
            <= period_end
        )
    ].copy()

    selected = selected.sort_values(
        [
            "event_date",
            "symbol",
        ]
    ).reset_index(
        drop=True
    )

    if selected.empty:

        raise ValueError(
            "No matched observations exist "
            "in the selected 7-year period."
        )

    print("\n=== SELECTED 7-YEAR PERIOD ===")
    print("Start year:", start_year)
    print("End year:", end_year)
    print("Companies:", selected["symbol"].nunique())
    print(selected["symbol"].value_counts())
    
    # ========================================================
    # 10. CREATE 80/20 CHRONOLOGICAL SPLIT
    # ========================================================
    #
    # Use trading dates rather than individual rows so that
    # the same date cannot appear partly in training and
    # partly in testing.
    # ========================================================

    selected_trading_dates = trading_dates[
        (trading_dates >= period_start)
        &
        (trading_dates <= period_end)
    ]

    split_index = int(
        len(selected_trading_dates)
        * TRAIN_RATIO
    )

    if (
        split_index <= 0
        or
        split_index >= len(
            selected_trading_dates
        )
    ):

        raise ValueError(
            "Invalid chronological split."
        )

    train_end_date = (
        selected_trading_dates[
            split_index - 1
        ]
    )

    test_start_date = (
        selected_trading_dates[
            split_index
        ]
    )

    train = selected.loc[
        selected["event_date"]
        <= train_end_date
    ].copy()

    test = selected.loc[
        selected["event_date"]
        >= test_start_date
    ].copy()


    # ========================================================
    # 11. PREVENT 3-DAY CAR TARGET LEAKAGE
    # ========================================================
    #
    # CAR(t) uses:
    #
    #     t
    #     t+1
    #     t+2
    #
    # All three trading days must stay inside the same
    # training/testing period.
    # ========================================================

    trading_position = {
        pd.Timestamp(date): position
        for position, date
        in enumerate(trading_dates)
    }


    def event_window_is_inside(
        event_date,
        period_first_date,
        period_last_date
    ):

        event_date = pd.Timestamp(
            event_date
        )

        if event_date not in trading_position:
            return False

        position = trading_position[
            event_date
        ]

        final_position = (
            position
            + EVENT_WINDOW_DAYS
            - 1
        )

        if final_position >= len(
            trading_dates
        ):
            return False

        event_window = trading_dates[
            position:
            final_position + 1
        ]

        if len(event_window) != EVENT_WINDOW_DAYS:
            return False

        return (
            event_window[0]
            >= period_first_date
            and
            event_window[-1]
            <= period_last_date
        )


    # --------------------------------------------------------
    # Training boundaries
    # --------------------------------------------------------

    train_first_date = (
        selected_trading_dates[0]
    )

    train_last_date = (
        train_end_date
    )

    train_mask = train[
        "event_date"
    ].apply(
        lambda date:
        event_window_is_inside(
            date,
            train_first_date,
            train_last_date
        )
    )

    train = train.loc[
        train_mask
    ].copy()


    # --------------------------------------------------------
    # Testing boundaries
    # --------------------------------------------------------

    test_first_date = (
        test_start_date
    )

    test_last_date = (
        selected_trading_dates[-1]
    )

    test_mask = test[
        "event_date"
    ].apply(
        lambda date:
        event_window_is_inside(
            date,
            test_first_date,
            test_last_date
        )
    )

    test = test.loc[
        test_mask
    ].copy()


    # Sort again after filtering.

    train = train.sort_values(
        [
            "event_date",
            "symbol",
        ]
    ).reset_index(
        drop=True
    )

    test = test.sort_values(
        [
            "event_date",
            "symbol",
        ]
    ).reset_index(
        drop=True
    )

    if train.empty:
        raise ValueError(
            "Training dataset is empty."
        )

    if test.empty:
        raise ValueError(
            "Testing dataset is empty."
        )


    # ========================================================
    # 12. DTM
    # ========================================================
    #
    # IMPORTANT:
    #
    # Fit ONLY on training documents.
    # ========================================================

    dtm_vectorizer = CountVectorizer()

    X_dtm_train = (
        dtm_vectorizer.fit_transform(
            train["news_text"]
        )
    )

    # Same fitted vectorizer.
    X_dtm_test = (
        dtm_vectorizer.transform(
            test["news_text"]
        )
    )


    # ========================================================
    # 13. TF-IDF
    # ========================================================
    #
    # Vocabulary and IDF are learned ONLY from training.
    # ========================================================

    tfidf_vectorizer = (
        TfidfVectorizer()
    )

    X_tfidf_train = (
        tfidf_vectorizer.fit_transform(
            train["news_text"]
        )
    )

    # Same fitted vectorizer.
    X_tfidf_test = (
        tfidf_vectorizer.transform(
            test["news_text"]
        )
    )


    # ========================================================
    # 14. CREATE OUTPUT DATASETS
    # ========================================================

    news_dtm_train = make_output(
        train,
        X_dtm_train
    )

    news_dtm_test = make_output(
        test,
        X_dtm_test
    )

    news_tfidf_train = make_output(
        train,
        X_tfidf_train
    )

    news_tfidf_test = make_output(
        test,
        X_tfidf_test
    )


    # ========================================================
    # 15. SAVE OUTPUT FILES
    # ========================================================

    news_dtm_train.to_csv(
        "news_dtm_train.csv",
        index=False
    )

    news_dtm_test.to_csv(
        "news_dtm_test.csv",
        index=False
    )

    news_tfidf_train.to_csv(
        "news_tfidf_train.csv",
        index=False
    )

    news_tfidf_test.to_csv(
        "news_tfidf_test.csv",
        index=False
    )


    # ========================================================
    # 16. SUMMARY
    # ========================================================

    print()
    print(
        "Selected 7-year period:"
    )

    print(
        f"{start_year}-01-01 "
        f"to {end_year}-12-31"
    )

    print()

    print(
        "Sampling method:"
    )

    print(
        "Random consecutive 7-year period "
        "using np.random.seed(42), followed "
        "by chronological 80/20 train/test split."
    )

    print()

    print(
        "Training period:"
    )

    print(
        train["event_date"].min().date(),
        "to",
        train["event_date"].max().date()
    )

    print()

    print(
        "Testing period:"
    )

    print(
        test["event_date"].min().date(),
        "to",
        test["event_date"].max().date()
    )

    print()

    print(
        f"Training observations: "
        f"{len(train):,}"
    )

    print(
        f"Testing observations: "
        f"{len(test):,}"
    )

    print()

    print(
        "DTM vocabulary size:",
        len(
            dtm_vectorizer.vocabulary_
        )
    )

    print(
        "TF-IDF vocabulary size:",
        len(
            tfidf_vectorizer.vocabulary_
        )
    )

    print()

    print(
        "Saved:"
    )

    print(
        "  news_dtm_train.csv"
    )

    print(
        "  news_dtm_test.csv"
    )

    print(
        "  news_tfidf_train.csv"
    )

    print(
        "  news_tfidf_test.csv"
    )


if __name__ == "__main__":
    main()