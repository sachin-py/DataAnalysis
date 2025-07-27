"""
Author: Sachin Kumar
GitHub: https://github.com/sachin-py/DataAnalysis/tree/main/NumPy

Description:
This script demonstrates essential data preprocessing techniques using NumPy, including:
- Handling missing values (using np.isnan, filling with mean or constant)
- Reshaping arrays
- Removing specific values, rows, or columns
- Sorting arrays in ascending and descending order
Each section illustrates typical preprocessing operations relevant in numerical and tabular data analysis.
"""
import numpy as np

# Checking for Missing Values

lending_co_data_numeric = np.loadtxt(
    "Lending-Company-Numeric-Data.csv", delimiter=',')
print("Original Data:\n", lending_co_data_numeric)

print("\nIs NaN present?\n", np.isnan(lending_co_data_numeric))
print("Total NaNs:", np.isnan(lending_co_data_numeric).sum())  # No missing values

lending_co_data_numeric_NAN = np.genfromtxt(
    "Lending-Company-Numeric-Data-NAN.csv", delimiter=';')
print("\nData with NaNs:\n", lending_co_data_numeric_NAN)
print("Is NaN present?\n", np.isnan(lending_co_data_numeric_NAN))
print("Total NaNs:", np.isnan(lending_co_data_numeric_NAN).sum())

# Filling missing values with 0
lending_co_data_numeric_NAN = np.genfromtxt(
    "Lending-Company-Numeric-Data-NAN.csv", delimiter=';', filling_values=0)
print("\nFilled with 0s:\n", lending_co_data_numeric_NAN)
print("Total NaNs after filling with 0:",
      np.isnan(lending_co_data_numeric_NAN).sum())

# Filling missing values with max value
lending_co_data_numeric_NAN = np.genfromtxt(
    "Lending-Company-Numeric-Data-NAN.csv", delimiter=';')
temporary_fill = np.nanmax(lending_co_data_numeric_NAN).round(2)
print("\nTemporary fill value (max):", temporary_fill)

lending_co_data_numeric_NAN = np.genfromtxt(
    "Lending-Company-Numeric-Data-NAN.csv", delimiter=';', filling_values=temporary_fill)
print("Total NaNs after filling with max value:",
      np.isnan(lending_co_data_numeric_NAN).sum())

# Substituting the Missing Values
lending_co_data_numeric_NAN = np.genfromtxt(
    "Lending-Company-Numeric-Data-NAN.csv", delimiter=';')
print("\nRaw data for substitution:\n", lending_co_data_numeric_NAN)
temporary_mean = np.nanmean(lending_co_data_numeric_NAN, axis=0).round(2)
print("Column-wise mean:\n", temporary_mean)

for i in range(lending_co_data_numeric_NAN.shape[1]):
    lending_co_data_numeric_NAN[:, i] = np.where(
        lending_co_data_numeric_NAN[:, i] == temporary_fill,
        temporary_mean[i],
        lending_co_data_numeric_NAN[:, i]
    )
print("Data after substituting NaNs with column means:\n",
      lending_co_data_numeric_NAN)

# Replace negative values with 0
for i in range(lending_co_data_numeric_NAN.shape[1]):
    lending_co_data_numeric_NAN[:, i] = np.where(
        lending_co_data_numeric_NAN[:, i] < 0,
        0,
        lending_co_data_numeric_NAN[:, i]
    )
print("Data after replacing negative values with 0:\n",
      lending_co_data_numeric_NAN)

# Reshaping Arrays
lending_co_data_numeric = np.loadtxt(
    "Lending-Company-Numeric-Data.csv", delimiter=',')
print("\nOriginal shape:", lending_co_data_numeric.shape)
print("Reshaped (6, 1043):\n", np.reshape(lending_co_data_numeric, (6, 1043)))
print("Transposed:\n", np.transpose(lending_co_data_numeric))
print("Reshaped (3, 2086):\n", np.reshape(lending_co_data_numeric, (3, 2086)))
print("Reshaped (2, 3, 1043):\n", np.reshape(
    lending_co_data_numeric, (2, 3, 1043)))

lending_co_data_numeric_2 = np.reshape(lending_co_data_numeric, (6, 1043))
print("lending_co_data_numeric_2:\n", lending_co_data_numeric_2)
print("Reshape (6, 1043) again:\n", lending_co_data_numeric.reshape(6, 1043))

# Removing Values
lending_co_data_numeric = np.loadtxt(
    "Lending-Company-Numeric-Data.csv", delimiter=',')
print("\nOriginal data for delete operations:\n", lending_co_data_numeric)

print("Shape after deleting first element:",
      np.delete(lending_co_data_numeric, 0).shape)
print("Total elements in original data:", lending_co_data_numeric.size)

print("Deleted first row:\n", np.delete(lending_co_data_numeric, 0, axis=0))
print("Deleted first column:\n", np.delete(lending_co_data_numeric, 0, axis=1))
print("Deleted second column:\n", np.delete(
    lending_co_data_numeric, 1, axis=1))
print("Deleted columns 0, 2, 4:\n", np.delete(
    lending_co_data_numeric, (0, 2, 4), axis=1))
print("Deleted columns (0,2,4) and rows (0,2,-1):\n",
      np.delete(np.delete(lending_co_data_numeric, (0, 2, 4), axis=1), [0, 2, -1], axis=0))

# Sorting Arrays
lending_co_data_numeric = np.loadtxt(
    "Lending-Company-Numeric-Data.csv", delimiter=',')
print("\nRow-wise sorted:\n", np.sort(lending_co_data_numeric))
print("Shape after sort (default):", np.sort(lending_co_data_numeric).shape)
print("Column-wise sort:\n", np.sort(lending_co_data_numeric, axis=0))

np.set_printoptions(suppress=True)
print("Column-wise sort with suppressed sci-notation:\n",
      np.sort(lending_co_data_numeric, axis=0))
print("Flattened and sorted:\n", np.sort(lending_co_data_numeric, axis=None))

# Descending order sort
print("Descending sorted:\n", -np.sort(-lending_co_data_numeric))

lending_co_data_numeric[:, 3] = np.sort(lending_co_data_numeric[:, 3])
print("4th column sorted:\n", lending_co_data_numeric[:, 3])

lending_co_data_numeric[:, 3].sort()
print("4th column sorted in-place:\n", lending_co_data_numeric[:, 3])

lending_co_data_numeric.sort(axis=0)
print("Full array sorted column-wise:\n", lending_co_data_numeric)
