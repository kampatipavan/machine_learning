import pandas as pd
import matplotlib.pyplot as plt
from io import StringIO

# ==========================================================
# 1. DATASET
# ==========================================================

sales_csv = """
Date,Product,Category,City,Quantity,Price
2026-01-01,Laptop,Electronics,Chennai,2,55000
2026-01-02,Mobile,Electronics,Madurai,5,25000
2026-01-03,Keyboard,Accessories,Chennai,10,3000
2026-01-04,Mouse,Accessories,Coimbatore,15,1500
2026-01-05,Laptop,Electronics,Salem,1,55000
2026-01-06,Mobile,Electronics,Chennai,3,25000
2026-01-07,Headphones,Accessories,Madurai,8,5000
2026-01-08,Keyboard,Accessories,Salem,12,3000
2026-01-09,Laptop,Electronics,Coimbatore,2,55000
2026-01-10,Mouse,Accessories,Chennai,20,1500
2026-01-11,Mobile,Electronics,Salem,4,25000
2026-01-12,Headphones,Accessories,Chennai,6,5000
"""

# ==========================================================
# 2. LOAD DATA
# ==========================================================

try:

    sales = pd.read_csv(
        StringIO(sales_csv)
    )

    print("Dataset loaded successfully!")

except Exception as e:

    print("Error while loading dataset:")
    print(e)


# ==========================================================
# 3. DISPLAY DATA
# ==========================================================

print("\n========== ORIGINAL DATA ==========")

print(sales)


# ==========================================================
# 4. CHECK DATA INFORMATION
# ==========================================================

print("\n========== DATA INFORMATION ==========")

print(sales.info())


# ==========================================================
# 5. CHECK MISSING VALUES
# ==========================================================

print("\n========== MISSING VALUES ==========")

print(sales.isnull().sum())


# ==========================================================
# 6. REMOVE DUPLICATES
# ==========================================================

print("\nDuplicate rows:",
      sales.duplicated().sum())

sales = sales.drop_duplicates()


# ==========================================================
# 7. CONVERT DATE COLUMN
# ==========================================================

sales["Date"] = pd.to_datetime(
    sales["Date"],
    errors="coerce"
)


# ==========================================================
# 8. CHECK INVALID DATES
# ==========================================================

invalid_dates = sales[
    sales["Date"].isnull()
]

print("\n========== INVALID DATES ==========")

print(invalid_dates)


# ==========================================================
# 9. CHECK NUMERIC COLUMNS
# ==========================================================

sales["Quantity"] = pd.to_numeric(
    sales["Quantity"],
    errors="coerce"
)

sales["Price"] = pd.to_numeric(
    sales["Price"],
    errors="coerce"
)


# ==========================================================
# 10. REMOVE INVALID NUMERIC RECORDS
# ==========================================================

sales = sales.dropna(
    subset=["Date", "Quantity", "Price"]
)


# ==========================================================
# 11. CREATE SALES COLUMN
# ==========================================================

sales["Sales"] = (
    sales["Quantity"] *
    sales["Price"]
)


print("\n========== DATA WITH SALES ==========")

print(sales)


# ==========================================================
# 12. USER-DEFINED FILTER
# ==========================================================

minimum_sales = 20000

filtered_data = sales[
    sales["Sales"] >= minimum_sales
]

print("\n========== FILTERED DATA ==========")

print(
    filtered_data[
        ["Product", "City", "Quantity", "Sales"]
    ]
)


# ==========================================================
# 13. GROUP BY CATEGORY
# ==========================================================

category_sales = sales.groupby(
    "Category"
)["Sales"].sum().sort_values(
    ascending=False
)

print("\n========== SALES BY CATEGORY ==========")

print(category_sales)


# ==========================================================
# 14. GROUP BY CITY
# ==========================================================

city_sales = sales.groupby(
    "City"
)["Sales"].sum().sort_values(
    ascending=False
)

print("\n========== SALES BY CITY ==========")

print(city_sales)


# ==========================================================
# 15. PRODUCT-WISE ANALYSIS
# ==========================================================

product_analysis = sales.groupby(
    "Product"
).agg(
    Total_Quantity=("Quantity", "sum"),
    Total_Sales=("Sales", "sum"),
    Average_Price=("Price", "mean")
).sort_values(
    by="Total_Sales",
    ascending=False
)

print("\n========== PRODUCT ANALYSIS ==========")

print(product_analysis)


# ==========================================================
# 16. TOP PRODUCTS
# ==========================================================

top_products = product_analysis.head(5)

print("\n========== TOP PRODUCTS ==========")

print(top_products)


# ==========================================================
# 17. SORT COMPLETE DATA
# ==========================================================

sorted_data = sales.sort_values(
    by="Sales",
    ascending=False
)

print("\n========== SORTED DATA ==========")

print(sorted_data)


# ==========================================================
# 18. RESET INDEX
# ==========================================================

sorted_data = sorted_data.reset_index(
    drop=True
)

print("\n========== RESET INDEX ==========")

print(sorted_data)


# ==========================================================
# 19. VISUALIZATION 1
#    SALES BY CITY
# ==========================================================

plt.figure(figsize=(8, 5))

city_sales.plot(
    kind="bar"
)

plt.title("Total Sales by City")
plt.xlabel("City")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()


# ==========================================================
# 20. VISUALIZATION 2
#    SALES BY CATEGORY
# ==========================================================

plt.figure(figsize=(8, 5))

category_sales.plot(
    kind="bar"
)

plt.title("Total Sales by Category")
plt.xlabel("Category")
plt.ylabel("Sales")
plt.xticks(rotation=0)
plt.tight_layout()

plt.show()


# ==========================================================
# 21. VISUALIZATION 3
#    PRODUCT SALES
# ==========================================================

plt.figure(figsize=(8, 5))

product_analysis["Total_Sales"].plot(
    kind="bar"
)

plt.title("Product-wise Sales")
plt.xlabel("Product")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()


# ==========================================================
# 22. FINAL VALIDATION
# ==========================================================

print("\n========== FINAL VALIDATION ==========")

print("Number of rows:",
      len(sales))

print("Number of columns:",
      len(sales.columns))

print(
    "Duplicate rows:",
    sales.duplicated().sum()
)

print(
    "Missing values:\n",
    sales.isnull().sum()
)

print("\nPipeline completed successfully!")