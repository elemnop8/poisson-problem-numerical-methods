""" This modul is for experimenting 
    Author: M. Nguyen, E. Tarielashvili.
    pylint Version 3.1.0
    pylint score: /10
"""

import numpy as np
import matplotlib.pyplot as plt
from block_matrix_2d import BlockMatrix
from linear_solvers import solve_lu
from poisson_problem_2d import rhs, compute_error, compute_error_plot

def get_u_function(kappa):
    def u_function(x1, x2):
        c = kappa * np.pi
        return x1 * np.sin(c * x1) * x2 * np.sin(c * x2)
    return u_function

def get_f(kappa):
    def f(x1, x2):
        c = kappa * np.pi
        cx1 = c * x1
        cx2 = c * x2

        x1_sin_cx1 = x1 * np.sin(cx1)
        x2_sin_cx2 = x2 * np.sin(cx2)

        return 2 * c * (c* x1_sin_cx1 * x2_sin_cx2 -
            x2_sin_cx2 * np.cos(cx1) -
            x1_sin_cx1 * np.cos(cx2)
        )
    return f

def main():
    """ Main program: testing imported moduls
    """
    kappa = 2
    n = 50

    f = get_f(kappa)
    u_function = get_u_function(kappa)

    b = rhs(n,f)

    p, l, u = BlockMatrix(n).get_lu()

    hat_u = solve_lu(p, l, u, b)

    error = compute_error(n, hat_u, u_function)

    print(f"For n={n} the error of the numerical solution of the Poisson problem is {error}")

    ns = range(4, n +1)
    N_values = [(n - 1)**2 for n in ns]
    error = np.empty(len(ns))
    #condition = np.empty(len(ns))

    for (i, n) in enumerate(ns):
        b = rhs(n,f)
        p, l, u = BlockMatrix(n).get_lu()
        hat_u = solve_lu(p, l, u, b)
        error[i] = compute_error(n, hat_u, u_function)
        #condition[i] = BlockMatrix(n).get_cond()

    #print(condition)

    plt.figure(figsize=(10, 6))
    plt.loglog(N_values, error, label="error")
    #plt.loglog(N_values, condition, label="condition of the matrix A")
    plt.xlabel("$N = (n-1)^2$", fontsize=15)
    plt.ylabel("$\|_\hat{u} - u \|_\infty$", fontsize=15)
    plt.title("error of the numerical solution of the Poisson problem", fontsize=16, pad=20)
    plt.legend(fontsize=13)
    plt.grid()
    #plt.savefig("error.png", dpi=600)
    plt.show()

if __name__ == '__main__':
    main()
