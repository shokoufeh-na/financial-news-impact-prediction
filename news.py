 


# -----------------------------Load the file headline.csv change timestamp one day--------------------------------# 


# The Safe Conservative Setup (Recommended)

# To completely eliminate the risk of look-ahead bias (trading on news that actually dropped after 4:00 PM), force the timestamp to 9:30 AM on the Next Trading Day
import pandas as pd

df = pd.read_csv("raw_partner_headlines_c2.csv", parse_dates=["date"])
# df = pd.read_csv("raw_partner_headlines_c2.csv")

print(df.head())
# If your original data is in UTC, localize to UTC first, then convert
df["date"] = df["date"].dt.tz_localize("UTC").dt.tz_convert("US/Eastern")

# Move to the next business day and assign 9:30 AM
df["date"] = (
    df["date"].dt.normalize()
    + pd.offsets.BDay(1)
    + pd.Timedelta(hours=9, minutes=30)
)
print(df.head())



###---------------------------------------------------  merg  2 files-----------------------------------------####


import pandas as pd

df1 = pd.read_csv("raw_analyst_ratings_c23.csv")
df2 = pd.read_csv("raw_partner_headlines_c22.csv")

print(df1.head())
print(df1.tail())
print(df2.head())
print(df2.tail())

# Merge the already-processed files
df = pd.concat([df1, df2], ignore_index=True)

# Preserve original/processed timestamp
df["timestamp"] = df["date"]

# Extract calendar date ONLY for grouping
df["date"] = df["timestamp"].astype(str).str[:10]

# Sort articles chronologically within each company-day
df = df.sort_values(["date", "symbol", "timestamp"])

# Preserve article-to-article alignment
for col in ["headline", "url", "publisher", "timestamp"]:
    df[col] = df[col].fillna("")

# Combine news for same company + same calendar date
 

SEPARATOR = "\x1f"
news = (
    df.groupby(["date", "symbol"], as_index=False)
      .agg({
          "headline": lambda x: SEPARATOR.join(x.astype(str)),
          "url": lambda x: SEPARATOR.join(x.astype(str)),
          "publisher": lambda x: SEPARATOR.join(x.astype(str)),
          "timestamp": lambda x: SEPARATOR.join(x.astype(str))
      })
)

# Sort final company-day rows
news = news.sort_values(
    ["date", "symbol"]
).reset_index(drop=True)

# Store the ordered timestamps in the date column
news["date"] = news["timestamp"]

# Temporary timestamp column is no longer needed
news = news.drop(columns=["timestamp"])

# Match required schema
news = news.rename(columns={"url": "URL"})

news = news[
    ["date", "symbol", "headline", "URL", "publisher"]
]

news.to_csv("news111.csv", index=False)

# Check result
df3 = pd.read_csv("news111.csv")

print(df3.head())
print(df3.tail())


#######............... change the timestamp to correct format eastern timezone for raw_analyst_ratings_c22.csv

import pandas as pd

df = pd.read_csv("raw_analyst_ratings_c22.csv")

print("--- Before Conversion ---")
print(df["date"].head())
print(df["date"].tail())


# 1. Split the logic to handle naive and aware timestamps properly
def standard_localize(date_series):
    # Coerce to string to avoid object type issues
    date_series = date_series.astype(str).str.strip()

    # Identify which rows are naive (don't contain a + or - offset near the end)
    is_naive = ~date_series.str.contains(r"[+-]\d{2}:?\d{2}$", regex=True)

    # Convert the naive rows, assuming they belong in US/Eastern, then make them strings again
    if is_naive.any():
        naive_dt = pd.to_datetime(date_series[is_naive], format="mixed")
        # Localize directly to Eastern without shifting the clock hours
        date_series[is_naive] = (
            naive_dt.dt.tz_localize("US/Eastern", ambiguous="NaT")
            .dt.strftime("%Y-%m-%d %H:%M:%S%z")
        )

    # Convert the already timezone-aware rows
    if (~is_naive).any():
        aware_dt = pd.to_datetime(
            date_series[~is_naive], format="mixed", utc=True
        )
        date_series[~is_naive] = (
            aware_dt.dt.tz_convert("US/Eastern")
            .dt.strftime("%Y-%m-%d %H:%M:%S%z")
        )

    return date_series


# 2. Run the function on your column
df["date"] = standard_localize(df["date"])

# 3. Standardize the final string offset format to guarantee the colon (e.g., -04:00)
df["date"] = df["date"].str.replace(
    r"([+-]\d{2})(\d{2})$", r"\1:\2", regex=True
)

 

df.to_csv("raw_analyst_ratings_c23.csv", index=False)



print("\n--- After Fixed Conversion ---")
print(df["date"].head())
print(df["date"].tail())


 
 