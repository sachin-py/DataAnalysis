
"""
Author: Sachin Kumar
GitHub: https: // github.com/sachin-py/DataAnalysis/tree/main/NumPy

This script demonstrates:
- Basic and stepwise slicing
- Conditional slicing using boolean indexing
- Understanding array dimensions and the squeeze function
"""
# ========================================
# ### Working with Arrays
# ========================================

import numpy as np

# ========================================
# #### Slicing
# array[ start_row:end_row:step_row , start_col:end_col:step_col ]
# ========================================
# - start_row:end_row:step_row specifies the slicing for rows.
# - start_col:end_col:step_col specifies the slicing for columns.
# - start, end, and step are optional.

matrix_A = np.array([[1, 2, 3], [4, 5, 6]])
print(matrix_A)

# ==== Basic Slicing ====
print(matrix_A[:])
print(matrix_A[0:0])
print(matrix_A[0:1])
print(matrix_A[0:2])
print(matrix_A[0:3])
print(matrix_A[:, :])
print(matrix_A[:1])
print(matrix_A[1:])
print(matrix_A[2:])
print(matrix_A[:2])
print(matrix_A[:-1])
print(matrix_A[:, 1:])      # columns from 1 to end
print(matrix_A[1:, 1:])     # rows from 1, columns from 1

# ==== Stepwise Slicing ====
matrix_B = np.array([
    [1, 2, 3, 4, 5],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15]
])
print(matrix_B)

print(matrix_B[::, ::])     # full array
print(matrix_B[::2, ::])    # every 2nd row
print(matrix_B[::, ::2])    # every 2nd column
print(matrix_B[::2, ::2])   # every 2nd row and column
print(matrix_B[::-1, ::])   # reverse rows
print(matrix_B[::-1, ::-1])  # reverse rows and columns

# ========================================
# ### Conditional Slicing (Boolean Indexing)
# ========================================

matrix_C = np.array([
    [1, 22, 23, 24, 25],
    [6, 7, 8, 9, 10],
    [11, 12, 13, 14, 15]
])
print(matrix_C)

print(matrix_C[:, 0])                     # first column
print(matrix_C[:, 0] > 2)                 # condition on column 0
print(matrix_C[:, :] > 2)                 # condition on full matrix
print(matrix_C[matrix_C[:, :] > 2])       # elements > 2
print(matrix_C[matrix_C[:, :] != 2])      # elements not equal to 2
print(matrix_C[matrix_C[:, :] % 2 == 0])  # even elements
print(matrix_C[(matrix_C % 2 == 0) & (matrix_C > 10)])  # even and >10
print(matrix_C[(matrix_C % 2 == 0) | (matrix_C > 10)])  # even or >10

# ========================================
# ### Dimensions and the squeeze() Function
# ========================================

matrix_D = np.array([
    [31, 22, 23, 24, 25],
    [6, 37, 8, 19, 10],
    [11, 12, 13, 14, 15]
])
print(matrix_D)

# Accessing an element
print(type(matrix_D[0, 0]))
print(matrix_D[0, 0])

# 1D and 2D slices
print(matrix_D[0, 0:1])          # 1D
print(type(matrix_D[0, 0:1]))
print(type(matrix_D[0:1, 0:1]))  # 2D
print(matrix_D[0:1, 0:1])

# Shape inspection
print(matrix_D[0, 0].shape)
print(matrix_D[0, 0:1].shape)
print(matrix_D[0:1, 0:1].shape)

# Squeezing dimensions
print(matrix_D[0:1, 0:1].squeeze())
print(matrix_D[0:1, 0:1].squeeze().shape)
print(np.squeeze(matrix_D[0:1, 0:1]))

# Shapes after squeeze on different slices
print(matrix_D[0, 0].squeeze().shape)
print(matrix_D[0, 0:1].squeeze().shape)
print(matrix_D[0:1, 0:1].squeeze().shape)
