import numpy as np

# 1. Gaussian Elimination and Backward Substitution
def gaussian_elimination_back_substitution(A, b):
    n = len(b)
    for i in range(n):
        for j in range(i + 1, n):
            if A[i][i] == 0:
                continue
            ratio = A[j][i] / A[i][i]
            for k in range(n):
                A[j][k] -= ratio * A[i][k]
            b[j] -= ratio * b[i]
    x = [0] * n
    for i in range(n - 1, -1, -1):
        sum_ax = sum(A[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (b[i] - sum_ax) / A[i][i]
    return x

# 2. LU Factorization (Doolittle’s method)
def lu_factorization(A):
    n = len(A)
    L = np.zeros((n, n))
    U = np.zeros((n, n))
    for i in range(n):
        L[i][i] = 1
        for j in range(i, n):
            U[i][j] = A[i][j] - sum(L[i][k] * U[k][j] for k in range(i))
        for j in range(i + 1, n):
            L[j][i] = (A[j][i] - sum(L[j][k] * U[k][i] for k in range(i))) / U[i][i]
    determinant = np.prod(np.diag(U))
    return L, U, determinant

# 3. Check if matrix is diagonally dominant
def is_diagonally_dominant(A):
    n = len(A)
    for i in range(n):
        diag = abs(A[i][i])
        off_diag_sum = sum(abs(A[i][j]) for j in range(n) if j != i)
        if diag < off_diag_sum:
            return False
    return True

# 4. Check if matrix is positive definite using leading principal minors
def is_positive_definite(A):
    n = len(A)
    for k in range(1, n + 1):
        sub_matrix = A[:k, :k]
        if np.linalg.det(sub_matrix) <= 0:
            return False
    return True

# ------------------------
# Input matrices and vectors

# Question 1
A1 = [[2, -1, 1],
      [1, 3, 1],
      [-1, 5, 4]]
b1 = [6, 0, -3]

# Question 2
A2 = [[1, 1, 0, 3],
      [2, 1, -1, 1],
      [3, -1, -1, 2],
      [-1, 2, 3, -1]]

# Question 3
A3 = [[9, 0, 5, 2, 1],
      [3, 9, 1, 2, 1],
      [0, 1, 7, 2, 3],
      [4, 2, 3, 12, 2],
      [3, 2, 4, 0, 8]]

# Question 4
A4 = np.array([[2, 2, 1],
               [2, 3, 0],
               [1, 0, 2]])

# ------------------------
# Run solutions

# Question 1
solution1 = gaussian_elimination_back_substitution(np.array(A1, dtype=float), np.array(b1, dtype=float))

# Question 2
L2, U2, det2 = lu_factorization(np.array(A2, dtype=float))

# Question 3
is_diag_dom = is_diagonally_dominant(A3)

# Question 4
is_pos_def = is_positive_definite(A4)

# ------------------------
# Print Results
print("1. Solution using Gaussian Elimination:")
for val in solution1:
    print(val)
print("\n", A1[0][:2])

print("\n2. LU Factorization:")
print("Determinant:", det2)
print("L matrix:\n", L2)
print("U matrix:\n", U2)

print("\n3. Diagonally Dominant:", "Yes" if is_diag_dom else "No")
print("\n4. Positive Definite:", "Yes" if is_pos_def else "No")

