# ABBA-GMRES
This Python toolbox consists of the itterative solvers AB- and BA-GMRES methods (ABBA methods). They are able to solve matched and unmatched normal equations arising from X-ray Computed Tomography.

The iterative GMRES methods use a sequence of matrix-vector products, and is therefore able to work with abstract user defined forward and backward projectors.

The ABBA GMRES methods have the following features
- Projectors can be dense/sparse matrices or abstract matrices
- The possibility to restart GMRES during iterations
- Automatic stopping criteria: Discrepency Principle and Normalized Cumulative Periodogram

## User defined operators
This package allows the user to provide its own forward and backward projectors.