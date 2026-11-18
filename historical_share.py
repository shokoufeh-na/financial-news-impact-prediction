import requests
import pandas as pd
import time

# ---------------------------------------------------------
# 4. Get yearly historical shares from SEC
# ---------------------------------------------------------



COMPANIES = {
    "NVDA": "NVIDIA Corporation",
    "JPM": "JPMorgan Chase & Co.",
    "JNJ": "Johnson & Johnson",
    "PG": "The Procter & Gamble Company",
    "KO": "The Coca-Cola Company",
    "CSCO": "Cisco Systems, Inc.",
    "KR": "The Kroger Co.",
    "FCX": "Freeport-McMoRan Inc.",
    "MRK": "Merck & Co., Inc.",
    "GILD": "Gilead Sciences, Inc."
}

START_YEAR = 2009
END_YEAR = 2020

HEADERS = {
    "User-Agent": "FinancialNewsResearch your_email@example.com",
    "Accept-Encoding": "gzip, deflate"
}

# Get SEC company ticker-to-CIK mapping
response = requests.get(
    "https://www.sec.gov/files/company_tickers.json",
    headers=HEADERS,
    timeout=30
)
response.raise_for_status()

ticker_data = response.json()

cik_lookup = {
    row["ticker"]: str(row["cik_str"]).zfill(10)
    for row in ticker_data.values()
}

# Initialize ONCE, outside the company loop
shares_lookup = {}

for symbol, company_name in COMPANIES.items():

    print(f"Retrieving shares for {symbol}...")

    cik = cik_lookup.get(symbol)

    if cik is None:
        print(f"CIK not found: {symbol}")
        continue

    url = (
        f"https://data.sec.gov/api/xbrl/"
        f"companyfacts/CIK{cik}.json"
    )

    response = requests.get(
        url,
        headers=HEADERS,
        timeout=60
    )
    response.raise_for_status()

    data = response.json()
    facts = data.get("facts", {})
    if symbol == "XOM":
        print("\nXOM share-related SEC tags:")

        for taxonomy, concepts in facts.items():
            for tag in concepts:
                if "sharesoutstanding" in tag.lower():
                    print(taxonomy, tag)
# ---------------------------------------------------------
# Extract shares from both SEC concepts
# ---------------------------------------------------------

    dei_shares = (
        facts.get("dei", {})
        .get("EntityCommonStockSharesOutstanding", {})
        .get("units", {})
        .get("shares", [])
    )

    gaap_shares = (
        facts.get("us-gaap", {})
        .get("CommonStockSharesOutstanding", {})
        .get("units", {})
        .get("shares", [])
    )

    share_facts = dei_shares + gaap_shares

    if symbol == "XOM":
        print("\nXOM SEC share data:")
        print("DEI observations:", len(dei_shares))
        print("US-GAAP observations:", len(gaap_shares))

        for tag, records in [
            ("DEI", dei_shares),
            ("US-GAAP", gaap_shares)
        ]:
            print(f"\n{tag} observations:")

            for item in records[:10]:
                print(
                    "end:", item.get("end"),
                    "form:", item.get("form"),
                    "filed:", item.get("filed"),
                    "value:", item.get("val")
                )

        print("\nObservations in 2009–2020:")
        for item in share_facts:
            if "2009" <= item.get("end", "")[:4] <= "2020":
                print(
                    item.get("end"),
                    item.get("form"),
                    item.get("val")
                )
                break

    if symbol == "XOM":
        print("DEI observations:", len(dei_shares))
        print("US-GAAP observations:", len(gaap_shares))

        for item in share_facts[:5]:
            print({
                "end": item.get("end"),
                "form": item.get("form"),
                "filed": item.get("filed"),
                "val": item.get("val")
            })


    if not share_facts:
        share_facts = (
            facts.get("us-gaap", {})
            .get("CommonStockSharesOutstanding", {})
            .get("units", {})
            .get("shares", [])
        )

    # Extract yearly shares for THIS company
    for year in range(START_YEAR, END_YEAR + 1):

        candidates = [
            item for item in share_facts
            if item.get("form") in ["10-K", "10-Q"]
            and item.get("end", "").startswith(str(year))
            and item.get("val") is not None
            and item.get("filed") is not None
        ]

        if candidates:
            latest_end = max(
                item["end"] for item in candidates
            )

            same_date = [
                item for item in candidates
                if item["end"] == latest_end
            ]

            selected = min(
                same_date,
                key=lambda x: x["filed"]
            )

            shares_lookup[(symbol, year)] = selected["val"]

        else:
            shares_lookup[(symbol, year)] = None
            print(f"Missing shares: {symbol}, {year}")

    time.sleep(0.25)


# ---------------------------------------------------------
# Save retrieved shares
# ---------------------------------------------------------

shares_rows = []

for (symbol, year), shares in shares_lookup.items():

    shares_rows.append({
        "symbol": symbol,
        "year": year,
        "shares_outstanding": shares
    })

shares_df = pd.DataFrame(shares_rows)

shares_df = shares_df.sort_values(
    ["symbol", "year"]
).reset_index(drop=True)

shares_df.to_csv(
    "historical_shares2.csv",
    index=False
)

print(shares_df)
print("Total rows:", len(shares_df))
print("Companies:", shares_df["symbol"].nunique())
print("Missing values:", shares_df["shares_outstanding"].isna().sum())
print("Saved historical_shares2.csv")
