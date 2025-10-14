# Numerical Solution of the 2D Poisson Problem 🧮

This project was developed as part of the *Project Practicum I* (2024).  
It implements and analyzes numerical methods for solving the 2D Poisson equation using **finite difference discretization** and **LU decomposition** in Python.

---

## 📘 Contents
- **Report:** `Bericht_Poisson.pdf`  
- **Code:**
  - `poisson_problem_2d.py` – setup of the Poisson equation
  - `linear_solvers.py` – LU decomposition and linear system solvers
  - `block_matrix_2d.py` – construction of block-diagonal matrices
  - `experiments_lu.py` – experimental evaluation and plotting

---

## ⚙️ Methods
- Finite difference discretization of the Laplace operator  
- Construction of sparse block matrices  
- LU decomposition with partial pivoting  
- Error analysis and condition number estimation

---

## 📊 Results
The numerical experiments demonstrate:
- Sparse matrix representations significantly reduce memory requirements  
- LU decomposition provides accurate results for moderate grid sizes  
- Second-order convergence $O(h^2)$ verified for the finite difference method

---

## ▶️ How to Run
### Requirements
- Python ≥ 3.9  
- Libraries: `numpy`, `scipy`, `matplotlib`

### Run experiments
```bash
python experiments/experiments_lu.py
