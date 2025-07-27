"""
Statistics with NumPy

Author: Sachin Kumar
GitHub: https://github.com/sachin-py/DataAnalysis/tree/main/NumPy

Description:
This script demonstrates various statistical and mathematical operations using NumPy, including:
 - Mean, min, max, and square root operations
 - Statistical order functions like peak-to-peak and percentiles
 - Average, weighted average, variance, and standard deviation
 - Correlation and covariance matrices
 - Histogram generation and visualization (1D, 2D, 3D)
 - NaN-safe statistical functions
"""

import matplotlib.pyplot as plt
from numpy.random import PCG64 as pcg
from numpy.random import Generator as gen
import numpy as np

# np.mean()

matrix_A = np.array([[1, 0, 0, 3, 1], [3, 6, 6, 2, 9], [4, 5, 8, 3, 0]])
print(matrix_A)

print(np.mean(matrix_A))                          # Mean of all elements
print(np.mean(matrix_A[0], dtype=np.float32))     # Mean of first row
print(np.mean(matrix_A[:, 0]))                     # Mean of first column
print(np.mean(matrix_A, axis=0))                  # Mean of each column
print(np.mean(matrix_A, axis=1))                  # Mean of each row
print(matrix_A.mean())                            # Mean using method
print(np.sqrt(matrix_A))                          # Square root of each element


# Min and Max Values

matrix_A = np.array([[1, 0, 0, 3, 1], [3, 6, 6, 2, 9], [4, 5, 8, 3, 0]])
print(matrix_A)

print(np.min(matrix_A))                           # Minimum of whole array
print(np.amin(matrix_A))                          # Same as above

print(np.minimum(matrix_A[0], matrix_A[2]))       # Element-wise minimum
print(np.minimum(matrix_A[1], matrix_A[2]))
print(np.minimum.reduce(matrix_A))                # Minimum across rows
print(np.min(matrix_A, axis=0))                   # Min of each column
print(np.max(matrix_A))                           # Maximum value
print(np.maximum.reduce(matrix_A))                # Maximum across rows


# Statistical Order Function

matrix_A = np.array([[1, 0, 0, 3, 1], [3, 6, 6, 2, 9], [4, 5, 8, 3, 0]])
print(matrix_A)

print(np.ptp(matrix_A))           # Peak to Peak: max - min of whole array
print(np.ptp(matrix_A, axis=0))   # Peak to Peak per column
print(np.ptp(matrix_A, axis=1))   # Peak to Peak per row

# Percentiles
print(np.sort(matrix_A, axis=None))
print(np.percentile(matrix_A, 70))
print(np.percentile(matrix_A, 70, interpolation="midpoint"))
print(np.percentile(matrix_A, 70, interpolation="lower"))
print(np.percentile(matrix_A, 70, interpolation="higher"))
print(np.percentile(matrix_A, 70, interpolation="nearest"))
print(np.percentile(matrix_A, 50))  # Median
print(np.percentile(matrix_A, 100))
print(np.percentile(matrix_A, 0))

# Quantile
print(np.quantile(matrix_A, 0.70))


# Averages and Variances

matrix_A = np.array([[1, 0, 0, 3, 1], [3, 6, 6, 2, 9], [4, 5, 8, 3, 0]])
print(matrix_A)

print(np.median(matrix_A))
print(np.sort(matrix_A, axis=None))
print(np.mean(matrix_A))
print(np.average(matrix_A))

# Weighted average
array_RG = gen(pcg(365))
array_weights = array_RG.random(size=(3, 5))
print(array_weights)
print(np.average(matrix_A, weights=array_weights))

print(np.var(matrix_A))           # Variance
print(np.std(matrix_A))           # Standard deviation


# Correlation and Covariance

matrix_A = np.array([[1, 0, 0, 3, 1], [3, 6, 6, 2, 9], [4, 5, 8, 3, 0]])
print(matrix_A)

print(np.cov(matrix_A))           # Covariance matrix
print(np.corrcoef(matrix_A))      # Correlation matrix


# Histograms

# Histograms count the frequency of values in bins
matrix_A = np.array([[1, 0, 0, 3, 1], [3, 6, 6, 2, 9], [4, 5, 3, 8, 0]])
print(matrix_A)

print(np.sort(matrix_A, axis=None))
print(np.histogram(matrix_A))                  # Default bins
print(np.histogram(matrix_A, bins=4))          # 4 bins
print(np.histogram(matrix_A, bins=4, range=(1, 7)))

# Visualize with matplotlib
plt.hist(matrix_A.flat, bins=np.histogram(matrix_A)[1])
plt.show()

# 2D Histogram
print(np.histogram2d(matrix_A[0], matrix_A[1], bins=4))

# 3D histogram using d-dimensional bins
print(np.histogramdd(matrix_A.transpose(), bins=4))


# NAN-Equivalent Functions

matrix_A = np.array([[1, 0, 0, 3, 1], [3, 6, 6, 2, 9], [4, 5, 3, 8, 0]])
print(matrix_A)

print(np.nanmean(matrix_A))         # Same as mean, as no NaNs exist
print(np.mean(matrix_A))

matrix_B = np.array([[1, 0, 0, 3, 1], [3, 6, np.nan, 2, 9], [4, 5, 3, 8, 0]])
print(matrix_B)

print(np.nanmean(matrix_B))         # Ignores NaN values
print(np.mean(matrix_B))            # Will return NaN
print(np.nanquantile(matrix_B, 0.7))
print(np.nanvar(matrix_B))          # Variance ignoring NaNs
