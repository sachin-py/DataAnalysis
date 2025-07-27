"""
Author : Sachin Kumar
GitHub : https://github.com/sachin-py/DataAnalysis/tree/main/NumPy

Description:
This script covers the fundamental features of NumPy including:
- Array creation and indexing
- Slicing operations
- Element-wise operations and broadcasting
- Assigning values
- Data types in NumPy
- Typecasting
- Universal functions
- Running operations over an axis
"""

import numpy as np

# -----------------------------
# Array Creation and Indexing
# -----------------------------

my_array = np.array([[1, 2, 3], [9, 8, 7]])

print("First row:", my_array[0])
print("Second row:", my_array[1])
print("First element of first row:", my_array[0][0])

print("All elements using nested loop:")
for row in my_array:
    for elem in row:
        print(elem)

print("Element at (1,1):", my_array[1][1])
print("Same element using comma notation:", my_array[1, 1])

# -----------------------------
# Slicing
# -----------------------------
print("Full array:", my_array[:])
print("First column:", my_array[:, 0])
print("From row 1 onwards:", my_array[1:, ])
print("Slice with row 1 col 1:", my_array[1:2, 1:2])

# Negative indexing on 1D array
array_1D = np.array([2, 5, 7])
print("Last element:", array_1D[-1])
print("Second last element:", array_1D[-2])
print("Third last element:", array_1D[-3])

# Negative indexing on 2D array
print("Last row:", my_array[-1])
print("Second last row:", my_array[-2])
print("Element using negative index:", my_array[-2][1], my_array[-2][-1])
print("Empty slice:", my_array[-1:-1, -2:-2])

# -----------------------------
# Complex Slicing Example
# -----------------------------
arr = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

print("Row 1 slice:\n", arr[1:2])
print("First two columns:\n", arr[:, 0:2])
print("Full copy:\n", arr[::, ::])
print("Every second row/column:\n", arr[::2, ::2])
print("Negative slicing:\n", arr[-3:-1, -4:-1:2])

# -----------------------------
# Assigning Values
# -----------------------------
print("Original array:\n", arr)

arr[1][2] = 345
arr[3, 1] = 1000
print("After updates:\n", arr)

arr[0] = 9
print("Set first row to 9:\n", arr)

arr[0] = [1, 2, 3, 4]
arr[0] = np.array([10, 20, 30, 40])
print("Set first row to list/array:\n", arr)

arr[0, :] = [1, 1, 1, 1]
arr[:, 0] = [2, 2, 2, 2]
print("Updated rows and columns:\n", arr)

arr[:] = 23
print("Entire array set to 23:\n", arr)

arr[0:2, 0:2] = [[1, 2], [3, 5]]
print("Top-left 2x2 updated:\n", arr)

arr[2:4, 2:4] = np.array([[7, 9], [7, 9]])
print("Bottom-right 2x2 updated:\n", arr)

# -----------------------------
# Elementwise Operations
# -----------------------------
array_1 = np.array([7, 8, 9])
array_2 = np.array([[1, 2, 3], [4, 5, 6]])

print("Add 2 to array_1:", array_1 + 2)
print("Add 3 to array_2:\n", array_2 + 3)
print("array_2 - array_1:\n", array_2 - array_1)
print("array_1 + array_2[0]:", array_1 + array_2[0])
print("array_1 * array_2:\n", array_1 * array_2)

array_3 = np.array([2, 3, 4])
print("array_1 * array_3:", array_1 * array_3)

# -----------------------------
# Supported DataTypes
# -----------------------------
array_1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
print("Float array:\n", array_1)

array_1 = np.array([[1, 2, 3], [4, 5, 6]], dtype="float32")
array_1 = np.array([[1, 2, 3], [4, 5, 6]], dtype="float16")
array_1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float16)
array_1 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.complex64)
print("Complex array:\n", array_1)

array_1 = np.array([[1, 0, 3], [234, 5, 6]], dtype=bool)
print("Boolean array:\n", array_1)

array_1 = np.array([[1, 0, 3], [234, 5, 6]], dtype="str")
print("String array:\n", array_1)

# -----------------------------
# Broadcasting
# -----------------------------
array_a = np.array([1, 2, 3])
array_b = np.array([[1], [2]])
matrix_c = np.array([[1, 2, 3], [4, 5, 6]])

print("array_a + array_b:\n", array_a + array_b)
print("array_b + matrix_c:\n", array_b + matrix_c)
print("np.add(array_a, matrix_c):\n", np.add(array_a, matrix_c))
print("np.add(array_b, matrix_c):\n", np.add(array_b, matrix_c))

# Broadcasting error
array_d = np.array([[1, 2], [2, 3]])
try:
    print(np.add(array_d, matrix_c))
except ValueError as e:
    print("Broadcasting error:", e)

# -----------------------------
# Typecasting
# -----------------------------
print("Addition with typecasting to float:\n",
      np.add(array_b, matrix_c, dtype=np.float64))

# -----------------------------
# Operations over Axis
# -----------------------------
print("Matrix:\n", matrix_c)
print("Column-wise mean (axis=0):", np.mean(matrix_c, axis=0))
print("Row-wise mean (axis=1):", np.mean(matrix_c, axis=1))
