#ifndef SYMNMF_H
#define SYMNMF_H
#include <stdio.h>

/* A struct to represent a 2D matrix */
typedef struct {
    double **entries;
    int num_rows;
    int num_cols;
} Matrix;

void free_matrix(Matrix *matrix);
void free_2d_array(double **arr, int num_rows);
Matrix *compute_transpose(Matrix *matrix);
Matrix *multiply_matrices(Matrix *matrix_A, Matrix *matrix_B);
Matrix *build_A(double **values, int num_points, int dim);
Matrix *build_D(double **values, int num_points, int dim);
Matrix *build_W(double **values, int num_points, int dim);
Matrix *run_symnmf_optimization(Matrix *H, Matrix *W);
double get_matrix_average(Matrix *matrix);
double** parse_input_file(FILE *file, int *num_points_out, int *dim_out);

#endif