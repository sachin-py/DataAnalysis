"""
Author : Sachin Kumar
GitHub : https://github.com/sachin-py/DataAnalysis/tree/main/NumPy

Description:
This script demonstrates basic NumPy array creation, properties like shape and type, and how different inputs behave.
"""

import numpy as np

# -----------------------------
# 1. Creating a 1D NumPy array
# -----------------------------
array_1D = np.array([1, 2, 3, 4, 5])
print("1D Array:")
print(array_1D)  # Output: [1 2 3 4 5]

# -----------------------------
# 2. Creating a 2D NumPy array
# -----------------------------
array_2D = np.array([[2, 3, 4], [7, 8, 9]])
print("\n2D Array:")
print(array_2D)
# Output:
# [[2 3 4]
#  [7 8 9]]

# -----------------------------
# 3. Checking types of arrays
# -----------------------------
print("\nType of array_1D:", type(array_1D))    # <class 'numpy.ndarray'>
print("Type of array_2D:", type(array_2D))      # <class 'numpy.ndarray'>

# -----------------------------
# 4. Checking shapes of arrays
# -----------------------------
print("\nShape of array_1D:", array_1D.shape)   # (5,)
print("Shape of array_2D:", array_2D.shape)     # (2, 3)

# Accessing dimensions
print("\nLength of array_1D (rows):", array_1D.shape[0])     # 5
print("Rows in array_2D:", array_2D.shape[0])                # 2
print("Columns in array_2D:", array_2D.shape[1])             # 3

# -----------------------------
# 5. Creating a scalar array (0D)
# -----------------------------
array_no = np.array(13)
print("\nScalar (0D) Array:")
print(array_no)                         # Output: 13
print("Type:", type(array_no))          # <class 'numpy.ndarray'>
print("Shape:", array_no.shape)         # Output: () - 0D

# -----------------------------
# 6. Creating a 1-element 1D array
# -----------------------------
array_d = np.array([15])
print("\n1-element 1D Array:")
print(array_d)                          # Output: [15]
print("Shape:", array_d.shape)          # Output: (1,)
