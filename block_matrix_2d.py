""" Module for constructing and analyzing sparse matrices
    designed for use in solving the Poisson problem on a unit square 
    using finite differences.
    Author: M. Nguyen, E. Tarielashvili.
    pylint Version 3.1.0
    pylint score: /10
"""
import numpy as np
from scipy.sparse import diags, block_diag, vstack, hstack, csr_matrix
from scipy.linalg import lu
import matplotlib.pyplot as plt

class BlockMatrix:
    """ Represents block matrices arising from finite difference approximations
    of the Laplace operator.

    Parameters
    ----------
    n : int
        Number of intervals in each dimension.

    Attributes
    ----------
    n : int
        Number of intervals in each dimension.

    Raises
    ------
    ValueError
        If n < 2.
    """

    def __init__(self, n):
        if n < 2:
            raise ValueError("n must be atleast 2.")
        self.n = n
        self.n_inner = n - 1
        self.N = (n - 1)**2  # Total number of unknowns.

    def get_sparse(self):
        """ Returns the block matrix as sparse matrix.

        Returns
        -------
        scipy.sparse.csr_matrix
            The block_matrix in a sparse data format.
        """
        # definition of matrix C
        C = diags([-1,4,-1],[-1,0,1],shape=(self.n_inner, self.n_inner)).toarray()

        # C as the diagonal of A
        A_diag = block_diag([C] * (self.n_inner), format="csr")

        # substraction of C with the side diagonals
        A = A_diag - diags([1, 1], [-self.n_inner, self.n_inner], shape=(self.N, self.N), format="csr")

        return A

    def eval_sparsity(self):
        """ Returns the absolute and relative numbers of non-zero elements of
        the matrix. The relative quantities are with respect to the total
        number of elements of the represented matrix.

        Returns
        -------
        int
            Number of non-zeros
        float
            Relative number of non-zeros
        """
        A = self.get_sparse()
        non_zeros = A.nnz #Number of nonzero matrix elements
        total_entries = self.N**2
        return non_zeros, non_zeros / total_entries

    def get_lu(self):
        """ Provides an LU-Decomposition of the represented matrix A of the
        form A = p * l * u

        Returns
        -------
        p : numpy.ndarray
            permutation matrix of LU-decomposition
        l : numpy.ndarray
            lower triangular unit diagonal matrix of LU-decomposition
        u : numpy.ndarray
            upper triangular matrix of LU-decomposition
        """
        A_dense = self.get_sparse().toarray()
        p, l, u = lu(A_dense)
        return p, l, u

    def eval_sparsity_lu(self):
        """ Returns the absolute and relative numbers of non-zero elements of
        the LU-Decomposition. The relative quantities are with respect to the
        total number of elements of the represented matrix.

        Returns
        -------
        int
            Number of non-zeros
        float
            Relative number of non-zeros
        """
        _, l, u = self.get_lu()
        non_zeros_l = np.count_nonzero(l) - self.N  # Exclude diagonal ones in L
        non_zeros_u = np.count_nonzero(u)
        total_non_zeros_lu = non_zeros_l + non_zeros_u
        total_entries_A = self.get_sparse().nnz
        return total_non_zeros_lu, total_non_zeros_lu / total_entries_A

    def plot_sparsity_N(self, max_n):
        """
        Plots the number of non-zero entries in A and its LU decomposition against N.

        Parameters
        ----------
        max_n : int
            Maximum value of n to consider.
        """
        ns = range(2, max_n + 1)
        N_values = [(n - 1)**2 for n in ns]
        non_zeros_A = []
        non_zeros_LU = []

        for n in ns:
            BM = BlockMatrix(n)
            non_zeros, _ = BM.eval_sparsity()
            non_zeros_lu, _ = BM.eval_sparsity_lu()
            non_zeros_A.append(non_zeros)
            non_zeros_LU.append(non_zeros_lu)

        plt.figure(figsize=(10, 6))
        plt.plot(N_values, non_zeros_A, label="$A$", marker='o')
        plt.plot(N_values, non_zeros_LU, label="$LU$ decomposition", marker='o')
        plt.xlabel("$N = (n-1)^2$", fontsize=15)
        plt.ylabel("Number of non-zero entries", fontsize=15)
        plt.title("Sparsity of A and LU decomposition", fontsize=16, pad=20)
        plt.legend(fontsize=12)
        plt.grid()
        plt.show()


    def get_cond(self):
        """ Computes the condition number of the represented matrix.

        Returns
        -------
        float
            condition number with respect to the infinity-norm
        """
        return np.linalg.cond(self.get_sparse().toarray())

def main():
    """ Main program: Use of the class implemented above 
        and all its functionalities
    """
    n = 3
        # instance of the BlockMatrix class
    BM = BlockMatrix(n)
    # Get the sparse matrix
    A = BM.get_sparse()
    absolute, relative = BM.eval_sparsity()
    print(f"For n={n} we define N:=({n}-1)^2 and the matrix A in " 
          "R^{N}x{N}. We demonstrate A as a spare-Matrix as followed: \n" 
          f"{A} ")
    print(f"The absolute number of non-zero elements of the matrix A are: {absolute}"
          "\n The relative numbers of non-zero elements of the matrix with respect"
          f" to the total number of elements of the represented matrix are: {relative}")
    
    # Plot sparsity
    max_n = 10  # Specify the maximum value for n
    BM.plot_sparsity_N(max_n)

# main-funktion
if __name__ == "__main__":
    main()