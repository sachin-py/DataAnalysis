"""
Author: Sachin Kumar
GitHub: https://github.com/sachin-py/DataAnalysis/tree/main/NumPy

This script demonstrates various ways to generate data using NumPy:
- Creating arrays with zeros, ones, full, empty
- Using _like variants
- Using arange and random generators
- Sampling from known distributions
- Exporting and importing random data
"""


# PCG is Permutation Congruential Generator
from numpy.random import PCG64 as pcg
from numpy.random import Generator as gen
import numpy as np


# ========================================
# ### np.empty(), np.zeros(), np.ones(), np.full()
# ========================================

array_empty = np.empty(shape=(2, 3))
print(array_empty)

array_0s = np.zeros(shape=(2, 3))
print(array_0s)

array_0s = np.zeros(shape=(2, 3), dtype=np.int16)
print(array_0s)

array_1s = np.ones(shape=(2, 3))
print(array_1s)

array_1s = np.ones(shape=(2, 3), dtype=np.int16)
print(array_1s)

array_full = np.full(shape=(2, 3), fill_value=2, dtype=np.int16)
print(array_full)

array_full = np.full(shape=(2, 3), fill_value='python')
print(array_full)

# ========================================
# ### _like functions
# np.empty_like, np.zeros_like, np.ones_like, np.full_like
# ========================================

matrix_A = np.array([[1, 0, 9, 8, 7],
                     [6, 5, 4, 8, 1],
                     [1, 2, 3, 4, 5]])
print(matrix_A)

matrix_A_empty_like = np.empty_like(matrix_A)
print(matrix_A_empty_like)

matrix_A_zero_like = np.zeros_like(matrix_A)
print(matrix_A_zero_like)

matrix_A_ones_like = np.ones_like(matrix_A)
print(matrix_A_ones_like)

matrix_A_full_like = np.full_like(matrix_A, fill_value=100)
print(matrix_A_full_like)

# ========================================
# ### np.arange() — generating non-random sequences
# ========================================

array_range = np.arange(10)
print(array_range)

array_range = np.arange(start=0, stop=20)
print(array_range)

array_range = np.arange(start=0, stop=20, step=2)
print(array_range)

array_range = np.arange(start=0, stop=20, step=2.5)
print(array_range)

array_range = np.arange(start=0, stop=20, step=2.5, dtype=np.float32)
print(array_range)

array_range = np.arange(start=0, stop=20, step=2.5, dtype=np.int32)
print(array_range)

# ========================================
# ### Random Generators and Seeds
# ========================================

array_RG = gen(pcg())
print(array_RG.normal())
print(array_RG.normal(size=5))
print(array_RG.normal(size=(5, 5)))

array_RG = gen(pcg(seed=365))
print(array_RG.normal(size=(5, 5)))
print(array_RG.normal(size=(5, 5)))

# ========================================
# ### Generating Integers, Probabilities and Random Choices
# ========================================

array_RG = gen(pcg(seed=365))
print(array_RG.integers(10, size=(5, 5)))

array_RG = gen(pcg(seed=365))
print(array_RG.integers(low=10, high=50, size=(5, 5)))

array_RG = gen(pcg(seed=365))
print(array_RG.random(size=(5, 5)))

array_RG = gen(pcg(seed=365))
print(array_RG.choice([1, 10, 19, 87, 0], size=(5, 5)))

# ========================================
# ### Generating Arrays from Known Distributions
# ========================================

array_RG = gen(pcg(seed=365))
print(array_RG.poisson(size=(5, 5)))

array_RG = gen(pcg(seed=365))
print(array_RG.poisson(lam=10, size=(5, 5)))

array_RG = gen(pcg(seed=365))
print(array_RG.binomial(n=100, p=0.4, size=(5, 5)))

array_RG = gen(pcg(seed=365))
print(array_RG.logistic(loc=9, scale=1.2, size=(5, 5)))

# ========================================
# ### Application of Random Generators
# ========================================

array_RG = gen(pcg(seed=365))
array_column_1 = array_RG.normal(loc=2, scale=3, size=1000)
array_column_2 = array_RG.normal(loc=7, scale=2, size=1000)
array_column_3 = array_RG.logistic(loc=11, scale=3, size=1000)
array_column_4 = array_RG.exponential(scale=4, size=1000)
array_column_5 = array_RG.geometric(p=0.7, size=1000)

print(array_column_2)

random_test_data = np.array([array_column_1,
                             array_column_2,
                             array_column_3,
                             array_column_4,
                             array_column_5])
print(random_test_data)
print(random_test_data.shape)

random_test_data = random_test_data.transpose()
print(random_test_data.shape)
print(random_test_data)

np.savetxt("Random-Test Generated from Numpy.csv",
           random_test_data, fmt="%s", delimiter=',')

loaded_data = np.genfromtxt(
    "Random-Test Generated from Numpy.csv", delimiter=',')
print(loaded_data)
