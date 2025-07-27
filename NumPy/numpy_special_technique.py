"""
--------------------------------------------------------------
Description:
This script demonstrates advanced NumPy operations commonly used in data preprocessing and analysis.
It includes examples of:
  - Sorting and indexing with argsort and sort
  - Finding condition-matching indices with np.argwhere
  - Handling missing data using np.isnan and filling strategies
  - Shuffling rows in a dataset using different random generators
  - Type casting between numeric and string formats
  - String stripping on structured text data
  - Array stacking: stack, vstack, hstack, and dstack
  - Array concatenation across various axes
  - Extracting unique values with counts and indices using np.unique


Datasets used:
  - Lending-Company-Numeric-Data.csv
  - Lending-Company-Numeric-Data-NAN.csv
  - Lending-Company-Total-Price.csv

Author: Sachin Kumar
GitHub: https://github.com/sachin-py/DataAnalysis/tree/main/NumPy
--------------------------------------------------------------

"""

import numpy as np
from numpy.random import shuffle, Generator as gen
from numpy.random import PCG64 as pcg

# -----------------------------
# Argument Sort
# -----------------------------

lending_co_data_numeric = np.loadtxt(
    "Lending-Company-Numeric-Data.csv", delimiter=',')
print(lending_co_data_numeric)

print(np.argsort(lending_co_data_numeric))
print(np.sort(lending_co_data_numeric))
print(np.argsort(lending_co_data_numeric, axis=0))
print(lending_co_data_numeric[482, 5])
print(np.argsort(lending_co_data_numeric[:, 0]))

lending_co_data_numeric = lending_co_data_numeric[np.argsort(
    lending_co_data_numeric[:, 0])]
np.set_printoptions(suppress=True)
print(lending_co_data_numeric)
print(lending_co_data_numeric.argsort(axis=0))

# -----------------------------
# np.argwhere
# -----------------------------

lending_co_data_numeric = np.loadtxt(
    "Lending-Company-Numeric-Data.csv", delimiter=',')
print(np.argwhere(lending_co_data_numeric))
print(np.argwhere(lending_co_data_numeric == False))
print(lending_co_data_numeric[116])
print(lending_co_data_numeric[430])
print(np.argwhere(lending_co_data_numeric < 1000))
print(np.argwhere(lending_co_data_numeric > 1000))
print(np.argwhere(lending_co_data_numeric % 2 == 0))
print(np.isnan(lending_co_data_numeric).sum())

lending_co_data_numeric_NAN = np.genfromtxt(
    "Lending-Company-Numeric-Data-NAN.csv", delimiter=';')
print(np.isnan(lending_co_data_numeric_NAN))
print(np.argwhere(np.isnan(lending_co_data_numeric_NAN)))
print(lending_co_data_numeric_NAN[11])
print(lending_co_data_numeric_NAN[152])

for array_index in np.argwhere(np.isnan(lending_co_data_numeric_NAN)):
    lending_co_data_numeric_NAN[array_index[0], array_index[1]] = 0

print(lending_co_data_numeric_NAN[11])
print(lending_co_data_numeric_NAN[152])
print(np.isnan(lending_co_data_numeric_NAN).sum())

# -----------------------------
# Shuffling Data
# -----------------------------

lending_co_data_numeric = np.loadtxt(
    "Lending-Company-Numeric-Data.csv", delimiter=',')[:8]
np.random.shuffle(lending_co_data_numeric)
print(lending_co_data_numeric)

lending_co_data_numeric = np.loadtxt(
    "Lending-Company-Numeric-Data.csv", delimiter=',')
shuffle(lending_co_data_numeric)
print(lending_co_data_numeric)

array_RG = gen(pcg())
array_RG.shuffle(lending_co_data_numeric)
print(lending_co_data_numeric)

# -----------------------------
# Type Casting
# -----------------------------

lending_co_data_numeric = np.loadtxt(
    "Lending-Company-Numeric-Data.csv", delimiter=',')
print(lending_co_data_numeric.astype(np.int32))

lending_co_data_numeric = lending_co_data_numeric.astype(str)
print(lending_co_data_numeric)
print(type(lending_co_data_numeric))

# print(lending_co_data_numeric.astype(np.int32)) --> str->float->int the right conversion technique
print(lending_co_data_numeric.astype(np.float32))

lending_co_data_numeric = lending_co_data_numeric.astype(np.float32)
print(lending_co_data_numeric.astype(np.int32))

print(lending_co_data_numeric)

lending_co_data_numeric = np.loadtxt(
    "Lending-Company-Numeric-Data.csv", delimiter=',')
print(lending_co_data_numeric.astype(np.float32).astype(np.int32))

# -----------------------------
# Stripping Text Data
# -----------------------------

lending_co_total_price = np.genfromtxt(
    "Lending-Company-Total-Price.csv",
    delimiter=',',
    dtype="str",
    skip_header=1,
    usecols=[1, 2, 4]
)

lending_co_total_price[:, 0] = np.chararray.strip(
    lending_co_total_price[:, 0], "id_")
lending_co_total_price[:, 1] = np.chararray.strip(
    lending_co_total_price[:, 1], "Product ")
lending_co_total_price[:, 2] = np.chararray.strip(
    lending_co_total_price[:, 2], "Location ")
print(lending_co_total_price)

# -----------------------------
# Stacking
# -----------------------------

lending_co_data_numeric = np.loadtxt(
    "Lending-Company-Numeric-Data.csv", delimiter=',')
lending_co_data_numeric_NAN = np.genfromtxt(
    "Lending-Company-Numeric-Data-NAN.csv", delimiter=';')

temporary_fill = np.nanmax(lending_co_data_numeric_NAN).round(2) + 1
temporary_mean = np.nanmax(lending_co_data_numeric_NAN, axis=0).round(2)

lending_co_data_numeric_NAN = np.genfromtxt(
    "Lending-Company-Numeric-Data-NAN.csv", delimiter=';', filling_values=temporary_fill)

for i in range(lending_co_data_numeric_NAN.shape[1]):
    lending_co_data_numeric_NAN[:, i] = np.where(
        lending_co_data_numeric_NAN[:, i] == temporary_fill,
        temporary_mean[i],
        lending_co_data_numeric_NAN[:, i]
    )

print(np.stack((lending_co_data_numeric[:, 0], lending_co_data_numeric[:, 1])))
print(np.stack((lending_co_data_numeric[:, 1], lending_co_data_numeric[:, 0])))
print(np.stack(
    (lending_co_data_numeric[:, 1], lending_co_data_numeric[:, 0]), axis=1))
print(np.stack(
    (lending_co_data_numeric[:, 1], lending_co_data_numeric[:, 0], lending_co_data_numeric[:, 2]), axis=1))

# -----------------------------
# vstack / hstack / dstack
# -----------------------------

print(np.vstack((lending_co_data_numeric, lending_co_data_numeric_NAN)))
print(np.vstack((lending_co_data_numeric, lending_co_data_numeric_NAN)).shape)

print(np.hstack((lending_co_data_numeric, lending_co_data_numeric_NAN)))
print(np.hstack((lending_co_data_numeric, lending_co_data_numeric_NAN)).shape)

print(np.dstack((lending_co_data_numeric, lending_co_data_numeric_NAN)))
print(np.dstack((lending_co_data_numeric, lending_co_data_numeric_NAN)).shape)
print(np.dstack((lending_co_data_numeric, lending_co_data_numeric_NAN))[0])
print(np.dstack((lending_co_data_numeric, lending_co_data_numeric_NAN))[0, 0])
print(np.dstack((lending_co_data_numeric,
      lending_co_data_numeric_NAN))[0, :, 0])

# -----------------------------
# Concatenation
# -----------------------------

lending_co_data_numeric = np.loadtxt(
    "Lending-Company-Numeric-Data.csv", delimiter=',')
print(np.concatenate(
    (lending_co_data_numeric[0, :], lending_co_data_numeric[1, :])))

lending_co_data_numeric_NAN = np.genfromtxt(
    "Lending-Company-Numeric-Data-NAN.csv", delimiter=';')
temporary_fill = np.nanmax(lending_co_data_numeric_NAN).round(2) + 1
temporary_mean = np.nanmax(lending_co_data_numeric_NAN, axis=0).round(2)

lending_co_data_numeric_NAN = np.genfromtxt(
    "Lending-Company-Numeric-Data-NAN.csv", delimiter=';', filling_values=temporary_fill)

for i in range(lending_co_data_numeric_NAN.shape[1]):
    lending_co_data_numeric_NAN[:, i] = np.where(
        lending_co_data_numeric_NAN[:, i] == temporary_fill,
        temporary_mean[i],
        lending_co_data_numeric_NAN[:, i]
    )

print(np.concatenate((lending_co_data_numeric, lending_co_data_numeric_NAN)))
print(np.concatenate((lending_co_data_numeric, lending_co_data_numeric_NAN)).shape)
print(np.concatenate((lending_co_data_numeric, lending_co_data_numeric_NAN), axis=1))
print(np.concatenate((lending_co_data_numeric,
      lending_co_data_numeric_NAN), axis=1).shape)
print(np.concatenate(
    (lending_co_data_numeric[0, :], lending_co_data_numeric[:, 0])))

# -----------------------------
# Unique Values
# -----------------------------

lending_co_data_numeric = np.loadtxt(
    "Lending-Company-Numeric-Data.csv", delimiter=',')
print(np.unique(lending_co_data_numeric))
print(np.unique(lending_co_data_numeric[:, 1]))
print(np.unique(lending_co_data_numeric[:, 1], return_counts=True))
print(np.unique(lending_co_data_numeric[:, 1],
      return_counts=True, return_index=True))
