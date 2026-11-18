"""
1_calculate_impacts_values.py

Calculate historical market-based impact values.

Input:
    historical_prices.csv

Output:
    historical_impacts_values.csv

Method:
- Daily return uses adjusted close.
- Volatility uses 60 trading days.
- Alpha/Beta use 120 trading days.
- A 20-trading-day gap separates estimation window from event date.
- Event window is t, t+1, t+2.
- CAR = AR(t) + AR(t+1) + AR(t+2).
- Z_CAR standardizes event CAR against historical CAR.
- IIS is based on Z_CAR.
- MIV = pre-event market capitalization * CAR.

The function assign_event_date() is also included so news timestamps
can later be mapped to valid trading dates using the same trading calendar.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONSTANTS
# ============================================================

INPUT_FILE = Path("historical_prices.csv")
OUTPUT_FILE = Path("historical_impacts_values-latest.csv")

MARKET_SYMBOL = "^sp500"

ESTIMATION_DAYS = 120
GAP_DAYS = 20
VOLATILITY_DAYS = 60
EVENT_WINDOW_DAYS = 3


OUTPUT_COLUMNS = [
    "date",
    "symbol",
    "open",
    "high",
    "low",
    "close",
    "adj_close",
    "volume",
    "market_cap",
    "daily_return",
    "daily_volatility",
    "market_return",
    "alpha",
    "beta",
    "AR",
    "CAR",
    "Z_CAR",
    "IIS",
    "MIV",
]


# ============================================================
# IIS
# ============================================================

def information_impact_score(z):
    """
    Convert Z_CAR to the assignment's IIS scale [-3, +3].
    """

    if pd.isna(z):
        return np.nan

    if z <= -3:
        return -3
    if z <= -2:
        return -2
    if z <= -1:
        return -1
    if z < 1:
        return 0
    if z < 2:
        return 1
    if z < 3:
        return 2

    return 3


# ============================================================
# MARKET MODEL
# ============================================================

def fit_market_model(
    stock_returns,
    market_returns
):
    """
    Estimate:

        R_stock = alpha + beta * R_market

    using exactly 120 valid trading-day observations.
    """

    paired = pd.concat(
        [
            pd.Series(
                stock_returns,
                dtype=float
            ),
            pd.Series(
                market_returns,
                dtype=float
            )
        ],
        axis=1
    ).dropna()

    # Require complete 120-day estimation window.
    if len(paired) != ESTIMATION_DAYS:
        return np.nan, np.nan

    stock = (
        paired.iloc[:, 0]
        .to_numpy()
    )

    market = (
        paired.iloc[:, 1]
        .to_numpy()
    )

    if not np.isfinite(stock).all():
        return np.nan, np.nan

    if not np.isfinite(market).all():
        return np.nan, np.nan

    if np.var(market) == 0:
        return np.nan, np.nan

    beta = (
        np.cov(
            market,
            stock,
            ddof=0
        )[0, 1]
        / np.var(market)
    )

    alpha = (
        stock.mean()
        - beta * market.mean()
    )

    return float(alpha), float(beta)


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # 1. LOAD HISTORICAL PRICE DATA
    # ========================================================

    if not INPUT_FILE.exists():

        raise FileNotFoundError(
            f"Cannot find {INPUT_FILE.resolve()}."
        )

    prices = pd.read_csv(
        INPUT_FILE
    )

    required_columns = {
        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "market_cap",
    }

    missing = (
        required_columns
        - set(prices.columns)
    )

    if missing:

        raise ValueError(
            "Missing columns: "
            f"{sorted(missing)}"
        )


    # ========================================================
    # 2. CLEAN DATA
    # ========================================================

    prices["date"] = (
        pd.to_datetime(
            prices["date"],
            errors="coerce"
        )
        .dt.normalize()
    )

    prices["symbol"] = (
        prices["symbol"]
        .astype(str)
        .str.strip()
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "market_cap",
    ]

    for column in numeric_columns:

        prices[column] = pd.to_numeric(
            prices[column],
            errors="coerce"
        )

    prices = prices.dropna(
        subset=[
            "date",
            "symbol",
            "adj_close"
        ]
    )

    prices = (
        prices
        .sort_values(
            ["symbol", "date"]
        )
        .drop_duplicates(
            ["date", "symbol"],
            keep="last"
        )
    )


    # ========================================================
    # 3. S&P 500 MARKET DATA
    # ========================================================

    market_rows = prices.loc[
        prices["symbol"].str.upper()
        == MARKET_SYMBOL.upper()
    ].copy()

    if market_rows.empty:

        raise ValueError(
            f"No {MARKET_SYMBOL} rows found."
        )

    market_rows = (
        market_rows
        .sort_values("date")
        .set_index("date")
    )

    # ========================================================
    # 4. ACTUAL TRADING CALENDAR
    # ========================================================
    #
    # Trading dates are obtained from the S&P 500 observations.
    # Weekends and market holidays are therefore absent.
    #
    # These dates are used for the estimation window,
    # 20-trading-day gap, and 3-trading-day CAR calculations.
    # ========================================================

    trading_dates = (
        pd.DatetimeIndex(
            market_rows.index
        )
        .normalize()
        .sort_values()
        .unique()
    )


    # ========================================================
    # 5. MARKET RETURN
    # ========================================================

    market_return = (
        market_rows["adj_close"]
        .pct_change(
            fill_method=None
        )
        .rename(
            "market_return"
        )
    )


    # ========================================================
    # 6. MARKET VOLATILITY
    # ========================================================

    market_volatility = (
        market_return
        .rolling(
            VOLATILITY_DAYS,
            min_periods=VOLATILITY_DAYS
        )
        .std()
    )


    result_parts = []


    # ========================================================
    # 7. KEEP S&P 500 IN OUTPUT
    # ========================================================

    index_out = (
        market_rows
        .reset_index()
    )

    index_out[
        "daily_return"
    ] = market_return.reindex(
        market_rows.index
    ).to_numpy()

    index_out[
        "daily_volatility"
    ] = market_volatility.reindex(
        market_rows.index
    ).to_numpy()

    index_out[
        "market_return"
    ] = index_out[
        "daily_return"
    ]

    for column in [
        "alpha",
        "beta",
        "AR",
        "CAR",
        "Z_CAR",
        "IIS",
        "MIV",
    ]:

        index_out[column] = np.nan

    result_parts.append(
        index_out
    )


    # ========================================================
    # 8. COMPANY DATA
    # ========================================================

    company_prices = prices.loc[
        prices["symbol"].str.upper()
        != MARKET_SYMBOL.upper()
    ]


    for symbol, group in company_prices.groupby(
        "symbol",
        sort=False
    ):

        group = (
            group
            .sort_values("date")
            .set_index("date")
        )


        # ====================================================
        # 9. ALIGN COMPANY TO ACTUAL TRADING DATES
        # ====================================================
        #
        # Reindex using the S&P 500 trading calendar.
        #
        # We do NOT forward-fill prices.
        # Missing company observations stay missing.
        # ====================================================

        joined = (
            group
            .reindex(trading_dates)
        )

        joined.index.name = "date"

        joined[
            "market_return"
        ] = market_return.reindex(
            trading_dates
        )


        # ====================================================
        # 10. STOCK DAILY RETURN
        # ====================================================

        joined[
            "daily_return"
        ] = (
            joined["adj_close"]
            .pct_change(
                fill_method=None
            )
        )


        # ====================================================
        # 11. 60-DAY VOLATILITY
        # ====================================================

        joined[
            "daily_volatility"
        ] = (
            joined["daily_return"]
            .rolling(
                VOLATILITY_DAYS,
                min_periods=VOLATILITY_DAYS
            )
            .std()
        )


        # Initialize impact columns.

        joined["alpha"] = np.nan
        joined["beta"] = np.nan
        joined["AR"] = np.nan
        joined["CAR"] = np.nan
        joined["Z_CAR"] = np.nan
        joined["IIS"] = np.nan
        joined["MIV"] = np.nan


        stock_return = joined[
            "daily_return"
        ]

        mkt_return = joined[
            "market_return"
        ]


        alpha_values = np.full(
            len(joined),
            np.nan
        )

        beta_values = np.full(
            len(joined),
            np.nan
        )

        ar_values = np.full(
            len(joined),
            np.nan
        )


        # ====================================================
        # 12. ESTIMATION WINDOW
        # ====================================================
        #
        # For every possible event trading date i:
        #
        #
        #      120 trading days
        # <----------------------->
        #
        #                         20 trading-day gap
        #                         <--------------->
        #
        # [ ESTIMATION WINDOW ]   [      GAP      ] EVENT
        #                                            |
        #                                            i
        #
        #
        # Example:
        #
        # event_idx = 200
        #
        # estimation_end
        #     = 200 - 20
        #     = 180
        #
        # estimation_start
        #     = 180 - 120
        #     = 60
        #
        # iloc[60:180]
        #
        # = exactly 120 trading observations.
        #
        # ====================================================

        for event_idx in range(
            len(joined)
        ):

            estimation_end = (
                event_idx
                - GAP_DAYS
            )

            estimation_start = (
                estimation_end
                - ESTIMATION_DAYS
            )

            if estimation_start < 0:
                continue


            estimation_stock = (
                stock_return.iloc[
                    estimation_start:
                    estimation_end
                ]
            )

            estimation_market = (
                mkt_return.iloc[
                    estimation_start:
                    estimation_end
                ]
            )


            # =================================================
            # 13. ALPHA / BETA
            # =================================================

            alpha, beta = fit_market_model(
                estimation_stock,
                estimation_market
            )

            if (
                pd.isna(alpha)
                or pd.isna(beta)
            ):
                continue


            alpha_values[
                event_idx
            ] = alpha

            beta_values[
                event_idx
            ] = beta


            # =================================================
            # 14. EVENT-DAY ABNORMAL RETURN
            # =================================================

            if (
                pd.notna(
                    stock_return.iloc[
                        event_idx
                    ]
                )
                and
                pd.notna(
                    mkt_return.iloc[
                        event_idx
                    ]
                )
            ):

                expected_return = (
                    alpha
                    + beta
                    * mkt_return.iloc[
                        event_idx
                    ]
                )

                ar_values[
                    event_idx
                ] = (
                    stock_return.iloc[
                        event_idx
                    ]
                    - expected_return
                )


        joined[
            "alpha"
        ] = alpha_values

        joined[
            "beta"
        ] = beta_values

        joined[
            "AR"
        ] = ar_values


        # ====================================================
        # 15. CAR
        #
        # CAR(t) =
        #
        # AR(t)
        # + AR(t+1)
        # + AR(t+2)
        #
        # ====================================================

        ar_series = pd.Series(
            ar_values,
            index=joined.index,
            dtype=float
        )

        car_values = (
            ar_series
            .rolling(
                EVENT_WINDOW_DAYS,
                min_periods=EVENT_WINDOW_DAYS
            )
            .sum()
            .shift(
                -(EVENT_WINDOW_DAYS - 1)
            )
        )

        joined[
            "CAR"
        ] = car_values


        # ====================================================
        # 16. Z_CAR
        # ====================================================

        z_values = np.full(
            len(joined),
            np.nan
        )

        car_array = (
            car_values
            .to_numpy(
                dtype=float
            )
        )


        for event_idx in range(
            len(joined)
        ):

            event_car = car_array[
                event_idx
            ]

            estimation_end = (
                event_idx
                - GAP_DAYS
            )

            estimation_start = (
                estimation_end
                - ESTIMATION_DAYS
            )


            if (
                estimation_start < 0
                or
                not np.isfinite(
                    event_car
                )
            ):
                continue


            alpha = alpha_values[
                event_idx
            ]

            beta = beta_values[
                event_idx
            ]


            if (
                not np.isfinite(alpha)
                or
                not np.isfinite(beta)
            ):
                continue


            # Historical AR using this event's
            # estimated alpha and beta.

            historical_ar = (
                stock_return.iloc[
                    estimation_start:
                    estimation_end
                ]
                -
                (
                    alpha
                    +
                    beta
                    *
                    mkt_return.iloc[
                        estimation_start:
                        estimation_end
                    ]
                )
            )


            # Historical 3-day CARs.

            historical_cars = (
                historical_ar
                .rolling(
                    EVENT_WINDOW_DAYS,
                    min_periods=EVENT_WINDOW_DAYS
                )
                .sum()
                .dropna()
            )


            if len(
                historical_cars
            ) < 2:
                continue


            mean_historical_car = (
                historical_cars.mean()
            )

            std_historical_car = (
                historical_cars.std(
                    ddof=1
                )
            )


            if (
                not np.isfinite(
                    std_historical_car
                )
                or
                std_historical_car
                <= 1e-12
            ):
                continue


            z_values[
                event_idx
            ] = (
                event_car
                - mean_historical_car
            ) / std_historical_car


        joined[
            "Z_CAR"
        ] = z_values


        # ====================================================
        # 17. IIS
        # ====================================================

        joined[
            "IIS"
        ] = pd.Series(
            z_values,
            index=joined.index
        ).map(
            information_impact_score
        )


        # ====================================================
        # 18. MIV
        #
        # Market cap immediately BEFORE event date.
        # ====================================================

        previous_market_cap = (
            joined[
                "market_cap"
            ]
            .shift(1)
        )

        joined[
            "MIV"
        ] = (
            previous_market_cap
            *
            joined["CAR"]
        )


        joined[
            "symbol"
        ] = symbol


        # Remove trading-calendar rows for which this company
        # has no historical price observation.

        joined = joined.loc[
            joined[
                "adj_close"
            ].notna()
        ]


        result_parts.append(
            joined.reset_index()
        )


    # ========================================================
    # 19. COMBINE OUTPUT
    # ========================================================

    output = pd.concat(
        result_parts,
        ignore_index=True,
        sort=False
    )


    # Ensure expected columns exist.

    for column in OUTPUT_COLUMNS:

        if column not in output.columns:
            output[column] = np.nan


    output = (
        output[
            OUTPUT_COLUMNS
        ]
        .sort_values(
            ["date", "symbol"]
        )
        .reset_index(
            drop=True
        )
    )


    # ========================================================
    # 20. SAVE
    # ========================================================

    output[
        "date"
    ] = pd.to_datetime(
        output["date"]
    ).dt.strftime(
        "%Y-%m-%d"
    )


    output.to_csv(
        OUTPUT_FILE,
        index=False,
        na_rep=""
    )


    print(
        f"Saved {len(output):,} rows "
        f"to {OUTPUT_FILE.resolve()}"
    )

    print(
        f"Trading dates: "
        f"{trading_dates.min().date()} "
        f"to "
        f"{trading_dates.max().date()}"
    )

    print(
        "Alpha/Beta: 120 trading-day estimation "
        "window with 20 trading-day gap."
    )

    print(
        "Event window: t, t+1, t+2."
    )

    print(
        "assign_event_date() is available for "
        "mapping news timestamps to this trading calendar."
    )


if __name__ == "__main__":
    main()