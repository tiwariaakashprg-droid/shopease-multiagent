import pandas as pd

df = pd.read_csv("data/test_queries.csv")

rows = []

for _, row in df.iterrows():

    q = str(row["query"])
    e = row["expected_policy"]

    # skip duplicate header
    if q == "query":
        continue

    # already correct row
    if pd.notna(e):
        rows.append({
            "query": q,
            "expected_policy": e
        })

    # broken row
    elif "\\t" in q or "\t" in q:

        if "\\t" in q:
            parts = q.split("\\t")
        else:
            parts = q.split("\t")

        if len(parts) == 2:
            rows.append({"query": parts[0].strip(),"expected_policy": parts[1].strip()})

        parts = q.split("\\t")

        if len(parts) == 2:
            rows.append({
                "query": parts[0].strip(),
                "expected_policy": parts[1].strip()
            })

clean = pd.DataFrame(rows)

# remove rows whose label still contains \t
clean = clean[
    ~clean["expected_policy"]
    .astype(str)
    .str.contains(r"\\t")
]

clean = clean.dropna()

print("Final shape:", clean.shape)

clean = clean[
    ~clean["expected_policy"]
    .astype(str)
    .str.contains(r"\\t", regex=True)
]

clean.to_csv(
    "data/test_queries_fixed.csv",
    index=False
)

print(clean.shape)
print(clean["expected_policy"].value_counts())
clean.to_csv("data/test_queries_fixed.csv", index=False)
print(clean.shape)