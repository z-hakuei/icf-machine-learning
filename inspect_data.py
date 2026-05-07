import pandas as pd
import numpy as np

xl = pd.ExcelFile('combined_data.xlsx')
print("Sheets:", xl.sheet_names)
for s in xl.sheet_names:
    df = pd.read_excel('combined_data.xlsx', sheet_name=s)
    print(f"\n=== Sheet: {s} | shape={df.shape} ===")
    print("Columns:", list(df.columns))
    print(df.head(3))
    print("dtypes:")
    print(df.dtypes)
    print("describe:")
    print(df.describe().T[['mean','std','min','max']])
