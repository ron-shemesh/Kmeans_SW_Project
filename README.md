Symmetric Non-negative Matrix Factorization (SymNMF) Clustering Project.
It is the final project in Software Project Course, TAU.

Overview
--------
This project implements the Symmetric Non-negative Matrix Factorization (SymNMF) clustering algorithm. 
It utilizes a hybrid architecture, leveraging the performance of C for the core numerical and matrix operations, and Python for the high-level control, data handling, and comparative analysis.

The project fulfills two main objectives:

1. Practical Implementation: Modular implementation of the SymNMF algorithm, including the calculation of the Similarity (A), Diagonal Degree (D), and Normalized Similarity (W) matrices, and the iterative optimization of the association matrix (H).

2. Comparative Analysis: Running both SymNMF and K-means on input datasets and comparing the clustering quality using the Silhouette Score from the scikit-learn library.

Architecture Details
--------------------

The system is divided into Python modules and a custom C extension, bridged by the Python C API.

**C Core (symnmf.c, symnmf.h, symnmfmodule.c):**
The C language is used to implement all performance-critical mathematical routines, including:

Building the A, D, and W matrices based on data points.

Matrix multiplication and transpose operations.

The iterative update rule for the H matrix until convergence or maximum iterations are reached.

The C API (symnmfmodule.c) handles the conversion of data between Python lists/floats and C Matrix structures.

**Python Interface (symnmf.py, analysis.py):**
The Python layer provides the user interface and controls the flow:

symnmf.py serves as the main command-line utility, calling the appropriate C function based on the requested goal ('symnmf', 'sym', 'ddg', or 'norm').

H Matrix Initialization: The initial H matrix is generated in Python using numpy.random.uniform() with a fixed seed (1234).

analysis.py performs the SymNMF to K-means comparison and calculates the final Silhouette Scores for reporting.

Numerical Constraints and Requirements
The following hard-coded constants are used throughout the implementation for convergence and output control:

Maximum Iterations (NMF & K-means): 300

Convergence Threshold (Epsilon): 
1e−4

NMF Update Beta: 0.5

Output Precision: All numerical outputs are formatted to 4 decimal places.


The project includes the following files, all contained within the submission directory:

symnmf.py (Python Interface)

symnmf.c (C Implementation of core routines)

symnmfmodule.c (Python C API Wrapper)

symnmf.h (C Header file)

analysis.py (Comparative Analysis and Evaluation)

setup.py (C Extension Build Script)

Makefile (C Executable Compilation Script)
