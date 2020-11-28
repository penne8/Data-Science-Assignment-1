import numpy as np

# consts
minimal_proximity = 1.00000000e-03
satisfy_equation_step = 0.01

def least_square(A, B):
    A_transpose = np.transpose(A)
    x = np.linalg.solve(A_transpose @ A, A_transpose @ B)
    return x

def minimal_loss_vector(A, x, B):
    curr_minimal_loss_vector =  (A @ x) - B
    return curr_minimal_loss_vector

def minimal_loss(r):
    return np.transpose(r) @ r

def sanity_check(A, r):
    numerical_zero = np.transpose(A) @ r
    negligible = True
    for value in numerical_zero:
        if value > minimal_proximity:
            negligible = False
    return negligible

def satisfy_equation(A, B, r, equation):
    W = np.eye(len(A))
    while np.absolute(r[equation]) >= minimal_proximity:
        W[equation, equation] += satisfy_equation_step
        A_transpose = np.transpose(A)
        x = np.linalg.solve(A_transpose @ W @ A, A_transpose @ W @ B)
        r = minimal_loss_vector(A, x, B)
    print(f"the 1st time that r_{equation + 1} is smaller then {minimal_proximity} is when W= ", W[equation, equation])
    print("sanity check passed: ", sanity_check(A, W @ r))
    return x

def main():

    A = np.asarray([[2, 1, 2], [1, -2, 1], [1, 2, 3], [1, 1, 1]])
    B = np.asarray([6, 1, 5, 2])

    # Finding LS solution
    least_square_solution = least_square(A, B)
    print("\nthis is the LS solution: \n", least_square_solution)

    # find minimal loss and validate
    curr_minimal_loss_vector = minimal_loss_vector(A, least_square_solution, B)
    print("the minimal loss (error) is: ", minimal_loss(curr_minimal_loss_vector))
    print("sanity check passed: ", sanity_check(A, curr_minimal_loss_vector))

    # find minimal r such that r < minimal_proximity
    least_square_solution = satisfy_equation(A, B, curr_minimal_loss_vector, 0)
    print("the least square solution is: ", least_square_solution)


if __name__ == '__main__':
    main()