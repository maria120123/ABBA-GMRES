# ABBA-GMRES
This Python toolbox consists of the itterative solvers AB- and BA-GMRES methods (ABBA methods) [1,2]. They are able to solve matched and unmatched normal equations arising from X-ray Computed Tomography.

The iterative GMRES methods use a sequence of matrix-vector products, and is therefore able to work with abstract user defined forward and backward projectors.

The ABBA GMRES methods have the following features
- Projectors can be dense/sparse matrices or abstract matrices
- The possibility to restart GMRES during iterations
- Automatic stopping criteria: Discrepency Principle [3] and Normalized Cumulative Periodogram [4]

## Package requirements
The following Python packages are required for the toolbox to work properly:
- [numpy](https://numpy.org/install/)
- [astra](https://github.com/astra-toolbox/astra-toolbox)
- [tigre](https://github.com/CERN/TIGRE)
- [GPUtil](https://pypi.org/project/GPUtil/)

## Pre-defined operators
We provide forward and back projectors from the public libraries ASTRA [5] and TIGRE [6]. Both packages has GPU implementations and will require a NVIDIA GPU, however, only ASTRA has a CPU implementation. 

### ASTRA projectors

The following ASTRA projectors are supported:
- _line_
- _strip_
- _linear_

However, depending on the setup- and device type, some of the projectors might not be supported. The following table highlights which types of operators are available for which setups.

|      | Line    | Strip   | Linear |
|:---- |:--------|:--------|:-------|
|    __Parallel beam__             ||
|CPU   | &check; | &check; | &check;|
|GPU   | &cross; | &cross; | &check;|
|    __Fan beam__                  ||
|CPU   | &check; | &check; | &cross;|
|GPU   | &cross; | &cross; | &check;|

The back projector is automatically chosen when using the ASTRA projectors. The resulting normal equations will be either matched or unmatched depending on the type of device used for computation, as highlighted by the following table. 

| Device | Parallel beam | Fan beam  |
|:----------|:--------------|:----------|
| CPU | Matched       | Matched   |    
| GPU | Unmatched     | Unmatched |

### TIGRE projectors
The following TIGRE forward projectors are supported:
- _Siddon_ (equivalent to line in ASTRA)
- _interpolated_ (equivalent to linear in ASTRA)

The following TIGRE backward projectors are supported:
- _matched_
- _FDK_

Choosing the FDK back projector will result in unmatched normal equations.

## User defined operators
This package allows the user to provide its own forward and backward projectors. The only requirement is that the user defines a custom class where the *\_\_matmul\_\_* function computes a matrix-vector product of the projector given an input vector __x__. 

```python
class custom_projector:
    # Compute matrix-vector product
    #   y = A*x
    def __matmul__(self,x):
        # Your code goes here
        # y = ...
        return y
```


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