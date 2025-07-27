"""
Author : Sachin Kumar
GitHub : https://github.com/sachin-py/DataAnalysis/tree/main/NumPy

This script demonstrates:
- Importing data from CSV and text files using NumPy
- Exporting data to text/CSV format using NumPy functions
- Using np.genfromtxt(), np.savetxt(), and related arguments for data handling
"""


# Importing Data

# Import required package
import numpy as np

# Loading data using np.loadtxt() - faster but breaks with missing or bad data
lending_co_data_numeric_1 = np.loadtxt(
    "Lending-Company-Numeric-Data.csv", delimiter=',')
print(lending_co_data_numeric_1)

# Loading data using np.genfromtxt() - slower but handles missing values
lending_co_data_numeric_2 = np.genfromtxt(
    "Lending-Company-Numeric-Data.csv", delimiter=',')
print(lending_co_data_numeric_2)

# Check if both arrays are equal
print(np.array_equal(lending_co_data_numeric_1, lending_co_data_numeric_2))

# Handling missing values using genfromtxt
lending_co_data_numeric_NAN = np.genfromtxt(
    "Lending-Company-Numeric-Data-NAN.csv", delimiter=';')
print(lending_co_data_numeric_NAN)

# If needed, load as string to avoid breaking
lending_co_data_numeric_NAN_str = np.loadtxt(
    "Lending-Company-Numeric-Data-NAN.csv", delimiter=';', dtype="str")
print(lending_co_data_numeric_NAN_str)

# Partial Cleaning While Importing

# Skip headers
lending_co_data_numeric_NAN = np.genfromtxt(
    "Lending-Company-Numeric-Data-NAN.csv", delimiter=';', skip_header=2)
print(lending_co_data_numeric_NAN)

# Skip footers
lending_co_data_numeric_NAN = np.genfromtxt(
    "Lending-Company-Numeric-Data-NAN.csv", delimiter=';', skip_footer=2)
print(lending_co_data_numeric_NAN)

# Skip both header and footer
lending_co_data_numeric_NAN = np.genfromtxt(
    "Lending-Company-Numeric-Data-NAN.csv", delimiter=';', skip_header=2, skip_footer=2)
print(lending_co_data_numeric_NAN)

# Use specific columns
lending_co_data_numeric_NAN = np.genfromtxt(
    "Lending-Company-Numeric-Data-NAN.csv", delimiter=';', usecols=(0,))
print(lending_co_data_numeric_NAN)

lending_co_data_numeric_NAN = np.genfromtxt(
    "Lending-Company-Numeric-Data-NAN.csv", delimiter=';', usecols=(0, 1, 5))
print(lending_co_data_numeric_NAN)

lending_co_data_numeric_NAN = np.genfromtxt(
    "Lending-Company-Numeric-Data-NAN.csv", delimiter=';', usecols=(5, 1, 0))
print(lending_co_data_numeric_NAN)

# Combine all arguments
lending_co_data_numeric_NAN = np.genfromtxt(
    "Lending-Company-Numeric-Data-NAN.csv", delimiter=';', usecols=(5, 1, 0), skip_header=2, skip_footer=2)
print(lending_co_data_numeric_NAN)

# Unpack columns into individual arrays
lending_co_data_5, lending_co_data_0, lending_co_data_1 = np.genfromtxt("Lending-Company-Numeric-Data-NAN.csv",
                                                                        delimiter=';',
                                                                        usecols=(
                                                                            5, 1, 0),
                                                                        skip_header=2,
                                                                        skip_footer=2,
                                                                        unpack=True)
print(lending_co_data_5)
print(lending_co_data_0)
print(lending_co_data_1)

# String vs Object vs Numbers

lending_co_lt = np.genfromtxt("lending-co-LT.csv", delimiter=',')
print(lending_co_lt)

lending_co_lt = np.genfromtxt(
    "lending-co-LT.csv", delimiter=',', dtype=np.int32)
print(lending_co_lt)

lending_co_lt = np.genfromtxt("lending-co-LT.csv", delimiter=',', dtype="str")
print(lending_co_lt)

lending_co_lt = np.genfromtxt(
    "lending-co-LT.csv", delimiter=',', dtype=np.object_)
print(lending_co_lt)

lending_co_lt = np.genfromtxt("lending-co-LT.csv", delimiter=',',
                              dtype=(np.int32, "str", "str", "str", "str", "str", np.int32))
print(lending_co_lt)

# Saving Data with NumPy

# Load data
lending_co = np.genfromtxt(
    "Lending-Company-Saving.csv", delimiter=',', dtype="str")
print(lending_co)

# Save as .npy (binary) file
np.save("Lending-company-saving-new", lending_co)

# Reload and verify
lending_data_save = np.load("Lending-company-saving-new.npy")
print(lending_data_save)
print(np.array_equal(lending_co, lending_data_save))

# Saving with .npz Format

# Save multiple arrays
np.savez("Lending-Company-Saving-NEWZ", lending_co, lending_data_save)

# Load .npz file
lending_data_save_load = np.load("Lending-Company-Saving-NEWZ.npz")
print(lending_data_save_load)
print(lending_data_save_load["arr_0"])
print(lending_data_save_load["arr_1"])

# Save with custom keys
np.savez("Lending-Company-Saving-NEWZ",
         company=lending_co, data_save=lending_data_save)

# Load and access custom keys
lending_data_savez = np.load("Lending-Company-Saving-NEWZ.npz")
print(lending_data_savez.files)
print(lending_data_savez['company'])
print(lending_data_savez['data_save'])

# Verify equality
print(np.array_equal(
    lending_data_savez['company'], lending_data_savez['data_save']))

# Save as .txt File

np.savetxt("Lending-Company-Saving-in-text",
           lending_co, fmt="%s", delimiter=',')
lending_data_saved_txt = np.genfromtxt(
    "Lending-Company-Saving-in-text", delimiter=',', dtype="str")
print(lending_data_saved_txt)

# Final check
print(np.array_equal(lending_data_saved_txt, lending_data_save))
