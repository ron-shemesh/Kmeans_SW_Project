#include "symnmf.h"
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>

#define BETA 0.5
#define EPSILON 0.0001
#define MAX_ITERATIONS 300

/* Helper function to create a new matrix struct and allocate memory for its entries.
Takes number of rows and columns as input, returns a pointer to the created matrix. */
static Matrix* create_matrix(int rows, int cols) {
    int i;
    Matrix *m = (Matrix*)malloc(sizeof(Matrix));
    if (m == NULL) {
        fprintf(stderr, "An Error Has Occurred\n");
        exit(1);
    }
    m->num_rows = rows;
    m->num_cols = cols;
    m->entries = (double**)calloc(rows, sizeof(double*));
    if (m->entries == NULL) {
        fprintf(stderr, "An Error Has Occurred\n");
        free(m);
        exit(1);
    }
    for (i = 0; i < rows; i++) {
        m->entries[i] = (double*)calloc(cols, sizeof(double));
        if (m->entries[i] == NULL) {
            fprintf(stderr, "An Error Has Occurred\n");
            /* Cleanup already allocated memory before exiting */
            free_matrix(m);
            exit(1);
        }
    }
    return m;
}

/* Calculates the squared Euclidean distance between two points.
Takes two point arrays and their dimension as input, returns the squared distance as double. */
static double calculate_squared_distance(const double *p1, const double *p2, int dim) {
    double sum = 0.0;
    int i;
    for (i = 0; i < dim; i++) {
        double diff = p1[i] - p2[i];
        sum += diff * diff;
    }
    return sum;
}

/* Calculates the sum of values in a specific row of a matrix. 
Takes a matrix pointer and the row index as input, returns the sum as double. */
static double sum_matrix_row(Matrix *matrix, int row_idx) {
    double sum = 0.0;
    int j;
    for (j = 0; j < matrix->num_cols; j++) {
        sum += matrix->entries[row_idx][j];
    }
    return sum;
}

/* Calculates the squared Frobenius norm of the difference between two matrices.
Takes two matrix pointers as input, returns the squared Frobenius norm as double. */
static double calculate_frobenius(Matrix *A, Matrix *B) {
    double norm_sum = 0.0;
    int i, j;
    for (i = 0; i < A->num_rows; i++) {
        for (j = 0; j < A->num_cols; j++) {
            double diff = A->entries[i][j] - B->entries[i][j];
            norm_sum += diff * diff;
        }
    }
    return norm_sum;
}

/* Checks if the optimization algorithm has converged or reached max iterations.
Takes the new and old H matrices and the current iteration count as input, returns 1 if converged, 0 otherwise. */
static int has_converged(Matrix *new_H, Matrix *old_H, int iter_count) {
    if (iter_count >= MAX_ITERATIONS) {
        return 1;
    }
    if (calculate_frobenius(new_H, old_H) < EPSILON) {
        return 1;
    }
    return 0;
}

/* Frees the memory allocated for a matrix.
Takes a matrix pointer as input. */
void free_matrix(Matrix *matrix) {
    int i;
    if (matrix != NULL) {
        if (matrix->entries != NULL) {
            for (i = 0; i < matrix->num_rows; i++) {
                free(matrix->entries[i]);
            }
            free(matrix->entries);
        }
        free(matrix);
    }
}

/* Frees the memory allocated for a 2D double array.
Takes a 2D array and its number of rows as input. */
void free_2d_array(double **arr, int num_rows) {
    int i;
    if (arr != NULL) {
        for (i = 0; i < num_rows; i++) {
            free(arr[i]);
        }
        free(arr);
    }
}

/* Computes the transpose of a given matrix.
Takes a matrix pointer as input, returns a new transposed matrix pointer. */
Matrix *compute_transpose(Matrix *matrix) {
    int i, j;
    Matrix *transposed = create_matrix(matrix->num_cols, matrix->num_rows);
    for (i = 0; i < transposed->num_rows; i++) {
        for (j = 0; j < transposed->num_cols; j++) {
            transposed->entries[i][j] = matrix->entries[j][i];
        }
    }
    return transposed;
}

/* Multiplies two matrices (A * B).
The loop order was changed from (i, j, k) to (i, k, j) , as it improves cache performance. 
This optimization was added after searching for a way to speed up the matrice multiplication(which was a slowing factor in the code).
Takes two matrix pointers as input, returns a new resulting matrix pointer. */
Matrix *multiply_matrices(Matrix *matrix_A, Matrix *matrix_B) {
    int i, j, k;
    Matrix *result_matrix;
    double r; 
    assert(matrix_A->num_cols == matrix_B->num_rows);
    result_matrix = create_matrix(matrix_A->num_rows, matrix_B->num_cols);

    for (i = 0; i < matrix_A->num_rows; i++) {
        for (k = 0; k < matrix_A->num_cols; k++) {
            r = matrix_A->entries[i][k];
            for (j = 0; j < matrix_B->num_cols; j++) {
                result_matrix->entries[i][j] += r * matrix_B->entries[k][j];
            }
        }
    }
    return result_matrix;
}


/* Executes a single H update iteration, following the rule in 1.4.2.
Takes W and H matrices as input, returns the updated H matrix. */
static Matrix* execute_H_update_step(Matrix *W, Matrix *H) {
    Matrix *WH, *H_transpose, *HHt, *HHtH, *new_H;
    int i, j;
    const double almost_zero = 1e-12;

    WH = multiply_matrices(W, H);
    H_transpose = compute_transpose(H);
    HHt = multiply_matrices(H, H_transpose);
    HHtH = multiply_matrices(HHt, H);
    new_H = create_matrix(H->num_rows, H->num_cols);

    for (i = 0; i < H->num_rows; i++) {
        for (j = 0; j < H->num_cols; j++) {
            double numerator = WH->entries[i][j];
            double denominator = HHtH->entries[i][j];
            if (denominator < almost_zero) { /* Prevent division by zero */
                denominator = almost_zero;
            }
            new_H->entries[i][j] = H->entries[i][j] * (1 - BETA + BETA * (numerator / denominator));
        }
    }
    /* Free temporary matrices */
    free_matrix(WH);
    free_matrix(H_transpose);
    free_matrix(HHt);
    free_matrix(HHtH);

    return new_H;
}


/* Builds the similarity matrix A, using symmetry to reduce calculations.
Takes a 2D array of point values, number of points, and dimension as input, returns the similarity matrix A. */
Matrix *build_A(double **values, int num_points, int dim) {
    int n = num_points;
    int i, j;
    Matrix *sim_matrix = create_matrix(n, n);

    for (i = 0; i < n; i++) {
        sim_matrix->entries[i][i] = 0.0;
        for (j = i + 1; j < n; j++) {
            double dist_sq = calculate_squared_distance(values[i], values[j], dim);
            double val = exp(-dist_sq / 2.0);
            sim_matrix->entries[i][j] = val;
            sim_matrix->entries[j][i] = val;
        }
    }
    return sim_matrix;
}


/* Builds the diagonal degree matrix D. 
This implementation doesn't use A, but recalculates distances, and is efficient in memory for the ddg goal, and it is not used in build_W.
Takes Points pointer as input, returns the degree matrix D. */
Matrix *build_D(double **values, int num_points, int dim) {
    int n = num_points;
    int i, j;
    Matrix *deg_matrix = create_matrix(n, n);

    for (i = 0; i < n; i++) {
        double row_sum = 0.0;
        for (j = 0; j < n; j++) {
            if (i != j) { /* recalculate distances only for non-diagonal elements */
                double dist_sq = calculate_squared_distance(values[i], values[j], dim);
                row_sum += exp(-dist_sq / 2.0);
            }
        }
        deg_matrix->entries[i][i] = row_sum;
    }
    return deg_matrix;
}

/* Builds the normalized similarity matrix W.
To achieve memory efficiency, it computes D's diagonal directly, to avoid storing the full D matrix,
this saves memory(we allocate an array instead of a matrix), while still doing the same calculations as build_D.
Takes Points pointer as input, returns the normalized similarity matrix W. */
Matrix *build_W(double **values, int num_points, int dim) {
    Matrix *A, *W;
    int n, i, j;
    double *D_diag_inv_sqrt; 

    A = build_A(values, num_points, dim);
    n = A->num_rows;
    D_diag_inv_sqrt = (double*)calloc(n, sizeof(double));
    assert(D_diag_inv_sqrt != NULL);
    
    for (i = 0; i < n; i++) {
        double row_sum = sum_matrix_row(A, i);
        if (row_sum > 0) {
            D_diag_inv_sqrt[i] = 1.0 / sqrt(row_sum);
        } else {
            D_diag_inv_sqrt[i] = 0.0;
        }
    }
    
    W = create_matrix(n, n);
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            W->entries[i][j] = D_diag_inv_sqrt[i] * A->entries[i][j] * D_diag_inv_sqrt[j];
        }
    }

    free_matrix(A);
    free(D_diag_inv_sqrt);
    return W;
}

/* Runs the SymNMF optimization algorithm.
It updates H iteratively until convergence or max iterations.
Takes initial H and W matrices as input, returns the optimized H matrix. */
Matrix *run_symnmf_optimization(Matrix *H, Matrix *W) {
    Matrix *H_curr, *H_next;
    int iter = 0;

    H_curr = H;
    while (1) {
        H_next = execute_H_update_step(W, H_curr);

        if (has_converged(H_next, H_curr, iter++)) {
            if (H_curr != H) {
                free_matrix(H_curr);
            }
            return H_next;
        }

        if (H_curr != H) {
            free_matrix(H_curr);
        }
        H_curr = H_next;
    }
}

/* Calculates the average of all elements in a matrix
Takes a matrix pointer as input, returns the average as double. */
double get_matrix_average(Matrix *matrix) {
    double total_sum = 0.0;
    int i;
    if (matrix == NULL || matrix->num_rows == 0 || matrix->num_cols == 0) {
        return 0.0;
    }
    for (i = 0; i < matrix->num_rows; i++) {
        total_sum += sum_matrix_row(matrix, i);
    }
    return total_sum / (matrix->num_rows * matrix->num_cols);
}

/* Reads input data from a file into a 2D array, dynamically allocating memory as needed.
The flat array and dynamic resizing are used to handle unknown input sizes.
Takes a file pointer and pointers to store number of points and dimension as input, returns the 2D array of point values. */
double** parse_input_file(FILE *file, int *num_points_out, int *dim_out) {
    double **values, *flat_data = NULL, val;
    long capacity = 10, count = 0;
    int num_points = 0, dim = 0, i, j, k; char sep;
    flat_data = (double*)malloc(capacity * sizeof(double));
    assert(flat_data != NULL);
/* Read all values into a flat array first */
    while (fscanf(file, "%lf%c", &val, &sep) == 2) {
        if (count >= capacity) {
            capacity *= 2;
            flat_data = (double*)realloc(flat_data, capacity * sizeof(double));
            assert(flat_data != NULL);
        }
        flat_data[count++] = val;
        if (sep == '\n') {
            num_points++;
            if (dim == 0) dim = (int)count;
        }
    }
    if (count > 0 && num_points == 0) { num_points = 1; dim = (int)count; }

    if (count == 0) {
        free(flat_data);
        *num_points_out = 0; *dim_out = 0;
        return NULL;
    }
/* converting the flat array to a 2D array */
    values = (double**)malloc(num_points * sizeof(double*));
    assert(values != NULL);
    for (i = 0, k = 0; i < num_points; i++) {
        values[i] = (double*)malloc(dim * sizeof(double));
        assert(values[i] != NULL);
        for (j = 0; j < dim; j++) values[i][j] = flat_data[k++];
    }

    free(flat_data);
    *num_points_out = num_points; *dim_out = dim;
    return values;
}


/* Prints the final matrix in the required format
Takes a matrix pointer as input. */
static void print_matrix_output(Matrix *matrix) {
    int i, j;
    for (i = 0; i < matrix->num_rows; i++) {
        for (j = 0; j < matrix->num_cols; j++) {
            printf("%.4f", matrix->entries[i][j]);
            if (j < matrix->num_cols - 1) {
                printf(",");
            }
        }
        printf("\n");
    }
}

/* Helper function to execute the required routine based on the given goal.
Takes the goal string, 2D array of point values, number of points, and dimension as input, returns the resulting matrix pointer. */
static Matrix* process_goal(const char *goal, double **values, int num_points, int dim) {
    if (strcmp(goal, "sym") == 0) {
        return build_A(values, num_points, dim);
    }
    if (strcmp(goal, "ddg") == 0) {
        return build_D(values, num_points, dim);
    }
    if (strcmp(goal, "norm") == 0) {
        return build_W(values, num_points, dim);
    }
    fprintf(stderr, "An Error Has Occurred\n");
    return NULL;
}

/* Main function for running the C program independently.
Takes command line arguments, processes the input file, executes the goal, and prints the result. */
int main(int argc, char *argv[]) {
    const char *goal, *filename;
    double **values;
    int num_points, dim;
    FILE *input_file;
    Matrix *result_matrix;

    if (argc != 3) {
        fprintf(stderr, "An Error Has Occurred\n");
        return 1;
    }
    goal = argv[1];
    filename = argv[2];
    input_file = fopen(filename, "r");

    if (input_file == NULL) {
        fprintf(stderr, "An Error Has Occurred\n");
        return 1;
    }

    values = parse_input_file(input_file, &num_points, &dim);
    fclose(input_file);
    result_matrix = process_goal(goal, values, num_points, dim);

    if (result_matrix == NULL) {
        free_2d_array(values, num_points);
        return 1;
    }

    print_matrix_output(result_matrix);
    free_matrix(result_matrix);
    free_2d_array(values, num_points);

    return 0;
}

