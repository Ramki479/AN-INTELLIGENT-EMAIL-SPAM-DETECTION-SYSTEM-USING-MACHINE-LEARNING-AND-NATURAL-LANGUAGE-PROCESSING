import pandas as pd
from DP import clean

# ---------- LOAD DATASET 1 ----------
df1 = pd.read_csv("raw_dataset.csv")

df1_final = pd.DataFrame({
    "Email": df1["text"],
    "Label": df1["label_num"]
})


# ---------- LOAD DATASET 2 ----------
df2 = pd.read_csv("messages.csv")

# Detect columns safely
text_col = "message" if "message" in df2.columns else "v2"
label_col = "label" if "label" in df2.columns else "v1"

# Normalize labels (VERY IMPORTANT)
df2[label_col] = df2[label_col].astype(str).str.lower().str.strip()

# Map labels
df2["Label"] = df2[label_col].map({
    "spam": 1,
    "ham": 0
})

df2_final = pd.DataFrame({
    "Email": df2[text_col],
    "Label": df2["Label"]
})


# ---------- COMBINE ----------
combined_df = pd.concat([df1_final, df2_final], ignore_index=True)

# ---------- REMOVE NaN LABELS ----------
combined_df.dropna(subset=["Label"], inplace=True)

# Ensure labels are integers
combined_df["Label"] = combined_df["Label"].astype(int)


# ---------- CLEAN TEXT ----------
cleaned_emails = []

for msg in combined_df["Email"]:
    cleaned_msg, _ = clean(str(msg))
    cleaned_emails.append(cleaned_msg)

combined_df["Email"] = cleaned_emails


# ---------- SAVE ----------
combined_df.to_csv("Cleaned_Data.csv", index=False)

print("✅ Cleaned_Data.csv created successfully")
print("Total rows after cleaning:", len(combined_df))
print("Label distribution:")
print(combined_df["Label"].value_counts())