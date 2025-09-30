#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "symnmf.h"

/* Converts a python list of lists into a C matrix struct. */
static Matrix* py_list_to_c_matrix(PyObject *py_list) {
    Py_ssize_t num_rows, num_cols, i, j;
    Matrix* c_matrix;

    if (!PyList_Check(py_list)) return NULL;
    num_rows = PyList_Size(py_list);
    if (num_rows == 0) return NULL;
    num_cols = PyList_Size(PyList_GetItem(py_list, 0));

    c_matrix = (Matrix*)malloc(sizeof(Matrix));
    if (!c_matrix) return NULL;

    c_matrix->entries = (double**)malloc(num_rows * sizeof(double*));
    if (!c_matrix->entries) {
        free(c_matrix);
        return NULL;
    }

    for (i = 0; i < num_rows; i++) {
        c_matrix->entries[i] = (double*)malloc(num_cols * sizeof(double));
        if (!c_matrix->entries[i]) {
            for (j = 0; j < i; j++) free(c_matrix->entries[j]);
            free(c_matrix->entries);
            free(c_matrix);
            return NULL;
        }
        for (j = 0; j < num_cols; j++) {
            PyObject *item = PyList_GetItem(PyList_GetItem(py_list, i), j);
            c_matrix->entries[i][j] = PyFloat_AsDouble(item);
        }
    }
    c_matrix->num_rows = (int)num_rows;
    c_matrix->num_cols = (int)num_cols;
    return c_matrix;
}

/* Converts a C matrix struct into a python list of lists. */
static PyObject* c_matrix_to_py_list(Matrix *c_matrix) {
    Py_ssize_t i, j;
    PyObject *py_list = PyList_New(c_matrix->num_rows);

    for (i = 0; i < c_matrix->num_rows; i++) {
        PyObject *row = PyList_New(c_matrix->num_cols);
        for (j = 0; j < c_matrix->num_cols; j++) {
            PyList_SetItem(row, j, PyFloat_FromDouble(c_matrix->entries[i][j]));
        }
        PyList_SetItem(py_list, i, row);
    }
    return py_list;
}

/* Converts a Python list of lists into a C 2D double array. */
static double** py_list_to_c_array(PyObject* py_list, int* num_points_out, int* dim_out) {
    double **c_array;
    Py_ssize_t num_points, dim, i, j;

    if (!PyList_Check(py_list)) return NULL;
    num_points = PyList_Size(py_list);
    if (num_points == 0) return NULL;
    dim = PyList_Size(PyList_GetItem(py_list, 0));

    c_array = (double**)malloc(num_points * sizeof(double*));
    if (!c_array) return NULL;

    for (i = 0; i < num_points; i++) {
        c_array[i] = (double*)malloc(dim * sizeof(double));
        if (!c_array[i]) {
            for (j = 0; j < i; j++) free(c_array[j]);
            free(c_array);
            return NULL;
        }
        for (j = 0; j < dim; j++) {
            PyObject *item = PyList_GetItem(PyList_GetItem(py_list, i), j);
            c_array[i][j] = PyFloat_AsDouble(item);
        }
    }
    *num_points_out = (int)num_points;
    *dim_out = (int)dim;
    return c_array;
}

/* Wrapper for the main SymNMF function. */
static PyObject* api_run_symnmf(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *h_py, *w_py, *result_py;
    Matrix *h_c, *w_c, *result_c;

    if (!PyArg_ParseTuple(args, "OO", &h_py, &w_py)) {
        PyErr_SetString(PyExc_TypeError, "An Error Has Occurred");
        return NULL;
    }

    h_c = py_list_to_c_matrix(h_py);
    w_c = py_list_to_c_matrix(w_py);
    if (!h_c || !w_c) {
        PyErr_SetString(PyExc_MemoryError, "An Error Has Occurred");
        return NULL;
    }

    result_c = run_symnmf_optimization(h_c, w_c);
    result_py = c_matrix_to_py_list(result_c);

    /* The original H (h_c) is freed by run_symnmf_optimization */
    free_matrix(w_c);
    free_matrix(result_c);

    return result_py;
}

/* Wrapper function to handle goals that take Points as input. */
static PyObject* api_calculate_matrix_from_points(PyObject *args, Matrix* (*goal_func)(double**, int, int)) {
    PyObject *py_list, *result_py;
    double **values_c;
    int num_points, dim;
    Matrix *result_c;

    if (!PyArg_ParseTuple(args, "O", &py_list)) {
        PyErr_SetString(PyExc_TypeError, "An Error Has Occurred");
        return NULL;
    }

    values_c = py_list_to_c_array(py_list, &num_points, &dim);
    if (!values_c) {
        PyErr_SetString(PyExc_MemoryError, "An Error Has Occurred");
        return NULL;
    }

    result_c = goal_func(values_c, num_points, dim);
    result_py = c_matrix_to_py_list(result_c);

    free_2d_array(values_c, num_points);
    free_matrix(result_c);

    return result_py;
}

/* Wrapper function for the 'sym' goal. */
static PyObject* api_sym(PyObject *self, PyObject *args) {
    (void)self;
    return api_calculate_matrix_from_points(args, build_A);
}

/* Wrapper function for the 'ddg' goal. */
static PyObject* api_ddg(PyObject *self, PyObject *args) {
    (void)self;
    return api_calculate_matrix_from_points(args, build_D);
}

/* Wrapper function for the 'norm' goal. */
static PyObject* api_norm(PyObject *self, PyObject *args) {
    (void)self;
    return api_calculate_matrix_from_points(args, build_W);
}

/* Wrapper function that calculates the average of a matrix. */
static PyObject* api_avg_mat(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *matrix_py;
    Matrix *matrix_c;
    double average;

    if (!PyArg_ParseTuple(args, "O", &matrix_py)) {
        PyErr_SetString(PyExc_TypeError, "An Error Has Occurred");
        return NULL;
    }

    matrix_c = py_list_to_c_matrix(matrix_py);
    if (!matrix_c) {
        PyErr_SetString(PyExc_MemoryError, "An Error Has Occurred");
        return NULL;
    }

    average = get_matrix_average(matrix_c);
    free_matrix(matrix_c);

    return PyFloat_FromDouble(average);
}


static PyMethodDef symnmf_methods[] = {
    {"sym", (PyCFunction)api_sym, METH_VARARGS, "Builds the similarity matrix."},
    {"ddg", (PyCFunction)api_ddg, METH_VARARGS, "Builds the diagonal degree matrix."},
    {"norm", (PyCFunction)api_norm, METH_VARARGS, "Builds the normalized similarity matrix."},
    {"avgMat", (PyCFunction)api_avg_mat, METH_VARARGS, "Calculates the average of a matrix."},
    {"SymNMF", (PyCFunction)api_run_symnmf, METH_VARARGS, "Runs the SymNMF optimization algorithm."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef symnmf_c_module = {
    PyModuleDef_HEAD_INIT,
    "symnmf_c",
    "A Python wrapper for the SymNMF algorithm implemented in C.",
    -1,
    symnmf_methods,
    NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit_symnmf_c(void) {
    return PyModule_Create(&symnmf_c_module);
}
