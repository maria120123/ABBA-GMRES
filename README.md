# ABBA-GMRES
This Python toolbox consists of the itterative solvers AB- and BA-GMRES methods (ABBA methods) [1,2]. They are able to solve matched and unmatched normal equations arising from X-ray Computed Tomography.

The iterative GMRES methods use a sequence of matrix-vector products, and is therefore able to work with abstract user defined forward and backward projectors.

The ABBA GMRES methods have the following features
- Projectors can be dense/sparse matrices or abstract matrices
- The possibility to restart GMRES during iterations
- Automatic stopping criteria: Discrepency Principle [3] and Normalized Cumulative Periodogram [4]

## Pre-defined operators
We provide operators from the public libraries ASTRA [5] and TIGRE [6]...


## User defined operators
This package allows the user to provide its own forward and backward projectors.




## Examples
We provide examples of each feature in the toolbox:
- _Example 1_: How to construct a CT problem with ASTRA and TIGRE.
- _Example 2_: How to use the ABBA methods.
- _Example 3_: How to use restart in the ABBA methods.
- _Example 4_: How to use the stopping rules in the ABBA methods.

## Citaitons
[1] Ken Hayami, et. al., _GMRES methods for least squares problems_
[2] Per Christian Hansen, et. al., _GMRES methods for tomographic reconstruction with an unmatched back projector_
[3] Per Christian Hansen, _Discrete Inverse Problems_
[4] Per Christian Hansen, _Implementation of the NCP method for CT problems_
[5] ASTRA: https://github.com/astra-toolbox/astra-toolbox
[6] TIGRE: https://github.com/CERN/TIGRE