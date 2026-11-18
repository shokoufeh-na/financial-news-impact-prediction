
import pandas as pd
import yfinance as yf
import numpy as np
import requests
import time


# =============================================================
# SEC SETTINGS
# =============================================================

# SEC asks API users to identify themselves.
# Replace this with your own email address.
# SEC_HEADERS = {
#     "User-Agent": "namazian@csu.fullerton.edu"
# }

# SEC_TICKER_URL = (
#     "https://www.sec.gov/files/company_tickers.json"
# )

# SEC_COMPANY_FACTS_URL = (
#     "https://data.sec.gov/api/xbrl/companyfacts/CIK{}.json"
# )


# =============================================================
# 1. Get SEC ticker -> CIK mapping
# =============================================================

# def get_sec_ticker_cik_map():

#     print("Downloading SEC ticker/CIK mapping...")

#     response = requests.get(
#         SEC_TICKER_URL,
#         headers=SEC_HEADERS,
#         timeout=30
#     )

#     response.raise_for_status()

#     data = response.json()

#     ticker_to_cik = {}

#     for item in data.values():

#         ticker = str(item["ticker"]).upper()

#         # SEC CIK must contain 10 digits
#         cik = str(item["cik_str"]).zfill(10)

#         ticker_to_cik[ticker] = cik

#     return ticker_to_cik


# # =============================================================
# # 2. Get annual historical shares outstanding from SEC
# # =============================================================

# def get_annual_historical_shares(
#     symbols,
#     start_year,
#     end_year
# ):

#     ticker_to_cik = get_sec_ticker_cik_map()

#     shares_lookup = {}

#     for symbol in symbols:

#         # SEC generally uses BRK-B rather than BRK.B
#         sec_symbol = symbol.replace(".", "-").upper()

#         cik = ticker_to_cik.get(sec_symbol)

#         if cik is None:
#             print(
#                 f"WARNING: SEC CIK not found for {symbol}"
#             )
#             continue

#         print(
#             f"Downloading historical shares for "
#             f"{symbol}..."
#         )

#         url = SEC_COMPANY_FACTS_URL.format(cik)

#         try:

#             response = requests.get(
#                 url,
#                 headers=SEC_HEADERS,
#                 timeout=30
#             )

#             response.raise_for_status()

#             company_data = response.json()

#         except Exception as e:

#             print(
#                 f"Failed SEC request for {symbol}: {e}"
#             )

#             continue


#         # -----------------------------------------------------
#         # Locate EntityCommonStockSharesOutstanding
#         # -----------------------------------------------------

#         try:

#             share_fact = (
#                 company_data["facts"]
#                 ["dei"]
#                 ["EntityCommonStockSharesOutstanding"]
#                 ["units"]
#                 ["shares"]
#             )

#         except KeyError:

#             print(
#                 f"No SEC shares-outstanding fact "
#                 f"found for {symbol}"
#             )

#             continue


#         # -----------------------------------------------------
#         # Convert SEC observations to DataFrame
#         # -----------------------------------------------------

#         records = []

#         for fact in share_fact:

#             if "val" not in fact:
#                 continue

#             if "end" not in fact:
#                 continue

#             end_date = pd.to_datetime(
#                 fact["end"],
#                 errors="coerce"
#             )

#             if pd.isna(end_date):
#                 continue

#             year = end_date.year

#             if year < start_year or year > end_year:
#                 continue

#             records.append({
#                 "year": year,
#                 "end": end_date,
#                 "shares": fact["val"],
#                 "form": fact.get("form"),
#                 "filed": fact.get("filed"),
#                 "fy": fact.get("fy"),
#                 "fp": fact.get("fp")
#             })


#         if not records:

#             print(
#                 f"No historical share observations "
#                 f"for {symbol}"
#             )

#             continue


#         facts_df = pd.DataFrame(records)


#         # -----------------------------------------------------
#         # Prefer annual 10-K observations
#         # -----------------------------------------------------

#         annual_df = facts_df[
#             facts_df["form"].isin(
#                 ["10-K", "10-K/A", "20-F", "20-F/A"]
#             )
#         ].copy()


#         # -----------------------------------------------------
#         # Select one observation for each calendar year
#         #
#         # We use the latest reported shares-outstanding
#         # observation associated with the annual filing.
#         # -----------------------------------------------------

#         for year in range(
#             start_year,
#             end_year + 1
#         ):

#             year_data = annual_df[
#                 annual_df["year"] == year
#             ].copy()

#             # If no annual filing observation is available,
#             # fall back to another SEC shares observation
#             # from that calendar year.
#             if year_data.empty:

#                 year_data = facts_df[
#                     facts_df["year"] == year
#                 ].copy()


#             if year_data.empty:

#                 print(
#                     f"  {symbol} {year}: "
#                     f"shares unavailable"
#                 )

#                 continue


#             # Convert filing date
#             year_data["filed_date"] = pd.to_datetime(
#                 year_data["filed"],
#                 errors="coerce"
#             )

#             # Sort so the latest observation is selected
#             year_data = year_data.sort_values(
#                 ["end", "filed_date"]
#             )

#             selected = year_data.iloc[-1]

#             shares = selected["shares"]

#             shares_lookup[
#                 (symbol, year)
#             ] = shares

#             print(
#                 f"  {symbol} {year}: "
#                 f"{shares:,.0f} shares"
#             )


#         # Be polite to SEC servers
#         time.sleep(0.15)


#     return shares_lookup

# =============================================================
# 3. Download historical prices and calculate market cap
# =============================================================

def download_historical_prices(start_date, end_date):

    # ---------------------------------------------------------
    # Load historical shares outstanding
    # ---------------------------------------------------------

    shares_df = pd.read_csv("historical_shares2.csv")

    shares_df["symbol"] = (
        shares_df["symbol"].astype(str).str.strip().str.upper()
    )

    shares_df["year"] = shares_df["year"].astype(int)

    shares_df["shares_outstanding"] = pd.to_numeric(
        shares_df["shares_outstanding"],
        errors="coerce"
    )

    # Get company tickers directly from historical_shares2.csv
    raw_tickers = shares_df["symbol"].dropna().unique().tolist()

    raw_tickers = [
        ticker for ticker in raw_tickers
        if ticker.lower() != "^sp500"
    ]

    # Check for missing or duplicate company-year records
    if shares_df.duplicated(["symbol", "year"]).any():
        raise ValueError("Duplicate company-year records found.")

    if shares_df["shares_outstanding"].isna().any():
        raise ValueError("Missing shares outstanding in CSV.")

    # Create lookup: (symbol, year) -> shares outstanding
    shares_lookup = {
        (row.symbol, row.year): row.shares_outstanding
        for row in shares_df.itertuples(index=False)
    }

    # Determine study years
    start_year = pd.Timestamp(start_date).year
    end_year = (
        pd.Timestamp(end_date) - pd.Timedelta(days=1)
    ).year

    print(f"\nStudy period: {start_year}-{end_year}")
    print("Companies:", raw_tickers)
    print("Number of companies:", len(raw_tickers))
    print("Missing shares:", shares_df["shares_outstanding"].isna().sum())

    # Verify that every company has shares for every study year
    missing_pairs = [
        (symbol, year)
        for symbol in raw_tickers
        for year in range(start_year, end_year + 1)
        if (symbol, year) not in shares_lookup
    ]

    if missing_pairs:
        raise ValueError(f"Missing company-year records: {missing_pairs}")



# def download_historical_prices(
#     sp500_csv_path,
#     start_date,
#     end_date
# ):

#     # ---------------------------------------------------------
#     # Load sp500.csv
#     # ---------------------------------------------------------

#     sp500_df = pd.read_csv(
#         sp500_csv_path
#     )


#     if "symbol" not in sp500_df.columns:

#         raise ValueError(
#             "sp500.csv must contain a 'symbol' column."
#         )


#     raw_tickers = (
#         sp500_df["symbol"]
#         .dropna()
#         .astype(str)
#         .str.strip()
#         .unique()
#         .tolist()
#     )


#     # Remove index if it somehow already exists
#     raw_tickers = [
#         t
#         for t in raw_tickers
#         if t.lower() != "^sp500"
#     ]


#     # ---------------------------------------------------------
#     # Determine study years
#     # ---------------------------------------------------------

#     start_year = pd.Timestamp(
#         start_date
#     ).year

#     # end date is exclusive for yfinance.
#     # Therefore subtract one day when determining final year.
#     end_year = (
#         pd.Timestamp(end_date)
#         - pd.Timedelta(days=1)
#     ).year


#     print(
#         f"\nStudy period: "
#         f"{start_year}-{end_year}"
#     )


    # # =========================================================
    # # 4. Get yearly historical shares from SEC
    # # =========================================================

    # print(
    #     "\nGetting annual historical "
    #     "shares outstanding from SEC...\n"
    # )


    # # shares_lookup = (
    # #     get_annual_historical_shares(
    # #         raw_tickers,
    # #         start_year,
    # #         end_year
    # #     )
    # # )
        
    # # =========================================================
    # # 4. Load yearly historical shares from CSV
    # # =========================================================

    # print("\nLoading historical shares from CSV...")

    # shares_df = pd.read_csv("historical_shares2.csv")

    # shares_df["symbol"] = (
    #     shares_df["symbol"].astype(str).str.strip().str.upper()
    # )

    # shares_df["year"] = shares_df["year"].astype(int)

    # shares_df["shares_outstanding"] = pd.to_numeric(
    #     shares_df["shares_outstanding"],
    #     errors="coerce"
    # )

    # # Create lookup: (symbol, year) -> shares outstanding
    # shares_lookup = {
    #     (row.symbol, row.year): row.shares_outstanding
    #     for row in shares_df.itertuples(index=False)
    # }

    # missing_symbols = set(raw_tickers) - set(shares_df["symbol"])

    # if missing_symbols:
    #     raise ValueError(
    #         f"Missing companies in historical_shares2.csv: {missing_symbols}"
    #     )

    # print("Companies:", shares_df["symbol"].nunique())
    # print("Years:", shares_df["year"].min(), "-", shares_df["year"].max())
    # print("Missing shares:", shares_df["shares_outstanding"].isna().sum())


    # # ---------------------------------------------------------
    # # Optional: save retrieved shares for inspection
    # # ---------------------------------------------------------

    # shares_rows = []

    # for (
    #     symbol,
    #     year
    # ), shares in shares_lookup.items():

    #     shares_rows.append({
    #         "symbol": symbol,
    #         "year": year,
    #         "shares_outstanding": shares
    #     })


    # shares_df = pd.DataFrame(
    #     shares_rows
    # )


    # if not shares_df.empty:

    #     shares_df = shares_df.sort_values(
    #         ["symbol", "year"]
    #     )

    #     shares_df.to_csv(
    #         "historical_shares.csv",
    #         index=False
    #     )

    #     print(
    #         "\nSaved retrieved annual shares to "
    #         "'historical_shares.csv'"
    #     )


    # =========================================================
    # 5. Prepare Yahoo Finance tickers
    # =========================================================

    yahoo_tickers = [
        ticker.replace(".", "-")
        for ticker in raw_tickers
    ]


    yahoo_index_ticker = "^GSPC"


    all_tickers = (
        yahoo_tickers
        + [yahoo_index_ticker]
    )


    print(
        f"\nDownloading historical prices "
        f"for {len(all_tickers)} assets..."
    )


    # =========================================================
    # 6. Download DAILY historical prices
    # =========================================================

    raw_data = yf.download(
        all_tickers,
        start=start_date,
        end=end_date,
        group_by="ticker",

        # IMPORTANT:
        # We need the actual Close price,
        # not an automatically adjusted price.
        auto_adjust=False,

        progress=True
    )


    # =========================================================
    # 7. Process each ticker
    # =========================================================

    all_rows = []


    for ticker in all_tickers:

        print(
            f"\nProcessing {ticker}..."
        )


        # -----------------------------------------------------
        # Extract ticker data
        # -----------------------------------------------------

        try:

            ticker_data = (
                raw_data[ticker]
                .dropna(how="all")
                .copy()
                .reset_index()
            )

        except KeyError:

            print(
                f"No Yahoo price data "
                f"for {ticker}"
            )

            continue


        # -----------------------------------------------------
        # Normalize column names
        # -----------------------------------------------------

        ticker_data.columns = [
            str(col)
            .lower()
            .replace(" ", "_")

            for col
            in ticker_data.columns
        ]


        # -----------------------------------------------------
        # Convert Yahoo symbol back to assignment symbol
        # -----------------------------------------------------

        if ticker == yahoo_index_ticker:

            output_symbol = "^sp500"

        else:

            output_symbol = (
                ticker.replace("-", ".")
            )


        # =====================================================
        # 8. Process each DAILY observation
        # =====================================================

        for _, row in ticker_data.iterrows():

            date = pd.Timestamp(
                row["date"]
            )

            year = date.year


            open_price = row.get(
                "open",
                np.nan
            )

            high_price = row.get(
                "high",
                np.nan
            )

            low_price = row.get(
                "low",
                np.nan
            )

            close_price = row.get(
                "close",
                np.nan
            )

            adj_close = row.get(
                "adj_close",
                close_price
            )

            volume = row.get(
                "volume",
                np.nan
            )


            # =================================================
            # 9. Calculate historical market capitalization
            # =================================================

            if output_symbol == "^sp500":

                # ---------------------------------------------
                # Assignment rule:
                #
                # S&P 500 is an INDEX, not an individual
                # company.
                #
                # Therefore market_cap MUST be NaN.
                # ---------------------------------------------

                market_cap = np.nan


            else:

                # ---------------------------------------------
                # Get historical shares for THIS company
                # and THIS year.
                # ---------------------------------------------

                historical_shares = (
                    shares_lookup.get(
                        (
                            output_symbol,
                            year
                        ),
                        np.nan
                    )
                )


                if (
                    pd.isna(
                        historical_shares
                    )
                    or
                    pd.isna(
                        close_price
                    )
                ):

                    market_cap = np.nan


                else:

                    # -----------------------------------------
                    # Assignment formula:
                    #
                    # Market Cap =
                    # Historical Shares Outstanding
                    # × Daily CLOSE Price
                    #
                    # DO NOT use adj_close here.
                    # -----------------------------------------

                    market_cap = (
                        historical_shares
                        * close_price
                    )
                    print(
                        f"Calculated market cap for {output_symbol} in {year}: {market_cap}"
                    )


            # =================================================
            # 10. Create output row
            # =================================================

            record = {

                "date":
                    date.strftime(
                        "%Y-%m-%d"
                    ),

                "symbol":
                    output_symbol,

                "open":
                    open_price,

                "high":
                    high_price,

                "low":
                    low_price,

                "close":
                    close_price,

                "adj_close":
                    adj_close,

                "volume":
                    volume,

                "market_cap":
                    market_cap
            }


            all_rows.append(
                record
            )


    # =========================================================
    # 11. Build historical_prices.csv
    # =========================================================

    if not all_rows:

        print(
            "No historical data "
            "was extracted."
        )

        return


    final_df = pd.DataFrame(
        all_rows
    )


    schema_cols = [

        "date",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "adj_close",
        "volume",
        "market_cap"
    ]


    final_df = final_df[
        schema_cols
    ]


    final_df = (
        final_df
        .sort_values(
            ["date", "symbol"]
        )
        .reset_index(
            drop=True
        )
    )


    # =========================================================
    # 12. Save final assignment file
    # =========================================================

    final_df.to_csv(
        "historical_prices.csv",
        index=False
    )


    print(
        "\nSuccessfully saved "
        "'historical_prices.csv'"
    )


    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    print(
        f"Rows: {len(final_df):,}"
    )


    print(
        "Date range:",
        final_df["date"].min(),
        "to",
        final_df["date"].max()
    )


    print(
        "\nMissing market_cap by symbol:"
    )


    print(
        final_df
        .groupby("symbol")
        ["market_cap"]
        .apply(
            lambda x:
            x.isna().sum()
        )
    )


# =============================================================
# RUN
# =============================================================

download_historical_prices(start_date="2009-01-01",

    # yfinance end date is EXCLUSIVE.
    # 2020-12-31 includes trading days through
    # Jan 1, 2021.
    end_date="2021-01-01"
)