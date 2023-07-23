import pandas as pd

# Load spreadsheet
xl = pd.ExcelFile('data.xls')

# Load a sheet into a DataFrame by its name
df = xl.parse(xl.sheet_names[0])

# Write to csv file
df.to_csv('data.csv', index=False)
