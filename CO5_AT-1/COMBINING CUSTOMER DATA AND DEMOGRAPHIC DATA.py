import pandas as pd

# ==========================================================
# 1. CREATE DATASET
# ==========================================================

# Customer Demographic Dataset
demographic_data = {
    "Customer_ID": [101, 102, 103, 104, 105, 106],
    "Name": ["Arun", "Bala", "Charan", "Divya", "Esha", "Fahad"],
    "Age": [22, 25, 30, 28, 35, 27],
    "City": ["Chennai", "Madurai", "Coimbatore",
             "Salem", "Trichy", "Chennai"]
}

demographics = pd.DataFrame(demographic_data)


# Customer Transaction Dataset
transaction_data = {
    "Customer_ID": [101, 102, 103, 104, 105, 107],
    "Product": ["Laptop", "Mobile", "Tablet",
                "Headphones", "Keyboard", "Mouse"],
    "Amount": [55000, 25000, 30000, 5000, 3000, 1500]
}

transactions = pd.DataFrame(transaction_data)


# ==========================================================
# 2. DISPLAY ORIGINAL DATA
# ==========================================================

print("========== DEMOGRAPHIC DATA ==========")
print(demographics)

print("\n========== TRANSACTION DATA ==========")
print(transactions)


# ==========================================================
# 3. CHECK DUPLICATE CUSTOMER IDs
# ==========================================================

print("\n========== DUPLICATE IDs ==========")

duplicate_demo = demographics[
    demographics["Customer_ID"].duplicated(keep=False)
]

duplicate_transaction = transactions[
    transactions["Customer_ID"].duplicated(keep=False)
]

print("Duplicate IDs in Demographics:")
print(duplicate_demo)

print("\nDuplicate IDs in Transactions:")
print(duplicate_transaction)


# ==========================================================
# 4. REMOVE DUPLICATES
# ==========================================================

demographics = demographics.drop_duplicates(
    subset="Customer_ID"
)

transactions = transactions.drop_duplicates(
    subset="Customer_ID"
)


# ==========================================================
# 5. FIND MISMATCHED CUSTOMER IDs
# ==========================================================

demo_ids = set(demographics["Customer_ID"])
transaction_ids = set(transactions["Customer_ID"])

# IDs available in transactions but not demographics
missing_in_demo = transaction_ids - demo_ids

# IDs available in demographics but not transactions
missing_in_transaction = demo_ids - transaction_ids

print("\n========== MISMATCHED IDs ==========")

print("IDs missing in Demographics:",
      missing_in_demo)

print("IDs missing in Transactions:",
      missing_in_transaction)


# ==========================================================
# 6. MERGE DATA
# ==========================================================

# Outer merge preserves ALL valid records
merged_data = pd.merge(
    demographics,
    transactions,
    on="Customer_ID",
    how="outer",
    indicator=True
)


# ==========================================================
# 7. CHECK MERGE STATUS
# ==========================================================

print("\n========== MERGE STATUS ==========")

print(
    merged_data["_merge"].value_counts()
)


# ==========================================================
# 8. FIND RECORDS WITH MISSING VALUES
# ==========================================================

print("\n========== RECORDS WITH MISSING VALUES ==========")

missing_records = merged_data[
    merged_data.isnull().any(axis=1)
]

print(missing_records)


# ==========================================================
# 9. GROUPING VALIDATION
# ==========================================================

print("\n========== CUSTOMER ID COUNT ==========")

id_count = merged_data.groupby(
    "Customer_ID"
).size()

print(id_count)


# Check IDs appearing more than once
duplicate_after_merge = id_count[
    id_count > 1
]

print("\nDuplicate IDs after merge:")
print(duplicate_after_merge)


# ==========================================================
# 10. REMOVE MERGE INDICATOR
# ==========================================================

merged_data = merged_data.drop(
    columns=["_merge"]
)


# ==========================================================
# 11. SORT DATA
# ==========================================================

merged_data = merged_data.sort_values(
    by="Customer_ID"
)


# ==========================================================
# 12. RESET INDEX
# ==========================================================

merged_data = merged_data.reset_index(
    drop=True
)


# ==========================================================
# 13. FINAL VALIDATION
# ==========================================================

print("\n========== FINAL DATA ==========")
print(merged_data)

print("\n========== FINAL INDEX ==========")
print(merged_data.index)

print("\n========== FINAL SHAPE ==========")
print(merged_data.shape)


# ==========================================================
# 14. CHECK FOR DUPLICATE IDs AGAIN
# ==========================================================

if merged_data["Customer_ID"].is_unique:
    print("\nSUCCESS: No duplicate Customer IDs.")
else:
    print("\nWARNING: Duplicate Customer IDs found!")


# ==========================================================
# 15. CHECK REQUIRED COLUMNS
# ==========================================================

required_columns = [
    "Customer_ID",
    "Name",
    "Age",
    "City",
    "Product",
    "Amount"
]

for column in required_columns:

    if column in merged_data.columns:
        print(f"{column}: Available")
    else:
        print(f"{column}: Missing")


# ==========================================================
# 16. FINAL SORTED OUTPUT
# ==========================================================

print("\n========== FINAL SORTED DATA ==========")

print(
    merged_data.sort_values(
        by=["City", "Customer_ID"]
    )
)