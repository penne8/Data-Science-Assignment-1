import numpy as np

A = np.asarray([[2, 1, 2], [1, -2, 1], [1, 2, 3], [1, 1, 1]])
b = np.asarray([6, 1, 5, 2])

# find least square solution
Atranspose = np.transpose(A)
ACoefficients = Atranspose @ A
BResults = Atranspose @ b
x = np.linalg.solve(ACoefficients, BResults)
print("the least square solution is: ",  x)

# find minimal loss
r = (A @ x) - b
minError = r @ np.transpose(r)
print("the minimal loss (error) is: ", minError)
zeroMat = Atranspose @ r
print("sanity check, should be close enough to 0: ", zeroMat)

# find minimal r such that r<10^-3
W = np.eye(4)
while np.absolute(r[0]) >= 1.00000000e-03:
    W[0, 0] = W[0, 0]+0.01
    ACoefficients = Atranspose @ W @ A
    BResults = Atranspose @ W @ b
    x = np.linalg.solve(ACoefficients, BResults)
    r = (A @ x) - b
print("the 1st time that r[1] is smaller then 10^-3 is when W= ", W[0, 0])
print("the least square solution is: ",  x)
minError = r @ np.transpose(r)
print("the minimal loss (error) is: ", minError)
zeroMat = Atranspose @ W @ r
print("sanity check, should be close enough to 0: ", zeroMat)