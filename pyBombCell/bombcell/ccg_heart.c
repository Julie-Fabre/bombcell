/* CCGHeart.c - Python C Extension Version
 * Adapted from MATLAB MEX version for Python
 * Fast cross-correlogram computation
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

#define PAIRBLOCKSIZE 100000000

typedef struct {
    unsigned int *pairs;
    unsigned int pair_cnt;
    unsigned int pair_sz;
} PairData;

static void add_pair(PairData *pd, unsigned int n1, unsigned int n2) {
    if (pd->pair_sz == 0) {
        pd->pairs = (unsigned int*)malloc(PAIRBLOCKSIZE * sizeof(unsigned int));
        pd->pair_sz = PAIRBLOCKSIZE;
        if (!pd->pairs) {
            PyErr_SetString(PyExc_MemoryError, "Could not allocate memory for pairs");
            return;
        }
    }
    
    if (pd->pair_cnt >= pd->pair_sz) {
        PyErr_SetString(PyExc_RuntimeError, "Too many pairs");
        return;
    }
    
    pd->pairs[pd->pair_cnt++] = n1;
    pd->pairs[pd->pair_cnt++] = n2;
}

static PyObject* ccg_heart(PyObject* self, PyObject* args) {
    PyArrayObject *times_array, *marks_array;
    double bin_size;
    unsigned int half_bins;
    int return_pairs = 0;
    
    // Parse arguments
    if (!PyArg_ParseTuple(args, "O!O!di|i", 
                         &PyArray_Type, &times_array,
                         &PyArray_Type, &marks_array, 
                         &bin_size, &half_bins, &return_pairs)) {
        return NULL;
    }
    
    // Check array types and shapes
    if (PyArray_TYPE(times_array) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_TypeError, "times must be double array");
        return NULL;
    }
    if (PyArray_TYPE(marks_array) != NPY_UINT32) {
        PyErr_SetString(PyExc_TypeError, "marks must be uint32 array");
        return NULL;
    }
    
    npy_intp n_spikes = PyArray_SIZE(times_array);
    if (PyArray_SIZE(marks_array) != n_spikes) {
        PyErr_SetString(PyExc_ValueError, "times and marks must have same length");
        return NULL;
    }
    
    // Get data pointers
    double *times = (double*)PyArray_DATA(times_array);
    unsigned int *marks = (unsigned int*)PyArray_DATA(marks_array);
    
    // Derive constants
    unsigned int n_bins = 1 + 2 * half_bins;
    double furthest_edge = bin_size * (half_bins + 0.5);
    
    // Count number of unique marks
    unsigned int n_marks = 0;
    for (npy_intp i = 0; i < n_spikes; i++) {
        if (marks[i] > n_marks) n_marks = marks[i];
        if (marks[i] == 0) {
            PyErr_SetString(PyExc_ValueError, "No zeros allowed in marks");
            return NULL;
        }
    }
    
    // Allocate output array
    npy_intp count_array_size = (npy_intp)n_marks * n_marks * n_bins;
    npy_intp dims[1] = {count_array_size};
    PyArrayObject *count_array = (PyArrayObject*)PyArray_ZEROS(1, dims, NPY_UINT32, 0);
    if (!count_array) {
        return NULL;
    }
    unsigned int *count = (unsigned int*)PyArray_DATA(count_array);
    
    // Initialize pair data if needed
    PairData pd = {NULL, 0, 0};
    
    // Main computation loop
    for (npy_intp center_spike = 0; center_spike < n_spikes; center_spike++) {
        unsigned int mark1 = marks[center_spike];
        double time1 = times[center_spike];
        
        // Go backward from center spike
        for (npy_intp second_spike = center_spike - 1; second_spike >= 0; second_spike--) {
            double time2 = times[second_spike];
            
            // Check if we've left the interesting region
            if (fabs(time1 - time2) > furthest_edge) break;
            
            // Calculate bin
            unsigned int bin = half_bins + (unsigned int)(floor(0.5 + (time2 - time1) / bin_size));
            
            unsigned int mark2 = marks[second_spike];
            npy_intp count_index = (npy_intp)n_bins * n_marks * (mark1 - 1) + 
                                  (npy_intp)n_bins * (mark2 - 1) + bin;
            
            // Bounds check
            if (count_index < 0 || count_index >= count_array_size) {
                char err_msg[256];
                snprintf(err_msg, sizeof(err_msg), 
                        "Index out of bounds: t1=%f t2=%f m1=%u m2=%u bin=%u index=%ld",
                        time1, time2, mark1, mark2, bin, (long)count_index);
                PyErr_SetString(PyExc_RuntimeError, err_msg);
                Py_DECREF(count_array);
                if (pd.pairs) free(pd.pairs);
                return NULL;
            }
            
            // Increment count
            count[count_index]++;
            if (return_pairs) {
                add_pair(&pd, (unsigned int)center_spike, (unsigned int)second_spike);
                if (PyErr_Occurred()) {
                    Py_DECREF(count_array);
                    if (pd.pairs) free(pd.pairs);
                    return NULL;
                }
            }
        }
        
        // Go forward from center spike
        for (npy_intp second_spike = center_spike + 1; second_spike < n_spikes; second_spike++) {
            double time2 = times[second_spike];
            
            // Check if we've left the interesting region
            if (fabs(time1 - time2) >= furthest_edge) break;
            
            // Calculate bin
            unsigned int bin = half_bins + (unsigned int)(floor(0.5 + (time2 - time1) / bin_size));
            
            unsigned int mark2 = marks[second_spike];
            npy_intp count_index = (npy_intp)n_bins * n_marks * (mark1 - 1) + 
                                  (npy_intp)n_bins * (mark2 - 1) + bin;
            
            // Bounds check
            if (count_index < 0 || count_index >= count_array_size) {
                char err_msg[256];
                snprintf(err_msg, sizeof(err_msg), 
                        "Index out of bounds: t1=%f t2=%f m1=%u m2=%u bin=%u index=%ld",
                        time1, time2, mark1, mark2, bin, (long)count_index);
                PyErr_SetString(PyExc_RuntimeError, err_msg);
                Py_DECREF(count_array);
                if (pd.pairs) free(pd.pairs);
                return NULL;
            }
            
            // Increment count
            count[count_index]++;
            if (return_pairs) {
                add_pair(&pd, (unsigned int)center_spike, (unsigned int)second_spike);
                if (PyErr_Occurred()) {
                    Py_DECREF(count_array);
                    if (pd.pairs) free(pd.pairs);
                    return NULL;
                }
            }
        }
    }
    
    // Return results
    if (return_pairs && pd.pair_cnt > 0) {
        npy_intp pair_dims[1] = {pd.pair_cnt};
        PyArrayObject *pair_array = (PyArrayObject*)PyArray_SimpleNew(1, pair_dims, NPY_UINT32);
        if (!pair_array) {
            Py_DECREF(count_array);
            free(pd.pairs);
            return NULL;
        }
        memcpy(PyArray_DATA(pair_array), pd.pairs, pd.pair_cnt * sizeof(unsigned int));
        free(pd.pairs);
        
        return Py_BuildValue("(NN)", count_array, pair_array);
    } else {
        if (pd.pairs) free(pd.pairs);
        return Py_BuildValue("(N)", count_array);
    }
}

static PyMethodDef CCGHeartMethods[] = {
    {"ccg_heart", ccg_heart, METH_VARARGS, 
     "Fast cross-correlogram computation\n"
     "Usage: ccg, [pairs] = ccg_heart(times, marks, bin_size, half_bins, [return_pairs])\n"
     "  times: array of spike times (double)\n"
     "  marks: array of unit IDs (uint32, no zeros)\n"
     "  bin_size: size of bins in time units (double)\n"
     "  half_bins: number of bins on each side of center (uint32)\n"
     "  return_pairs: optional, return spike pairs if True (default False)\n"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef ccg_heart_module = {
    PyModuleDef_HEAD_INIT,
    "ccg_heart",
    "Fast cross-correlogram computation using C",
    -1,
    CCGHeartMethods
};

PyMODINIT_FUNC PyInit_ccg_heart(void) {
    import_array();
    return PyModule_Create(&ccg_heart_module);
}