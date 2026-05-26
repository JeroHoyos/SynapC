#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

#include "tensor.h"

int* compute_strides(const int* shape,int ndim) {

    int* strides = malloc(ndim * sizeof(int));

    for(int i = 0; i < ndim; i++) {
        strides[i] = 1;
    }

    for(int i = ndim - 2; i >= 0; i--) {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    return strides;
}

Tensor tensor_init(float* data,int data_size,const int* shape,int ndim ) {

    int expected_size = 1;

    for(int i = 0; i < ndim; i++) {
        expected_size *= shape[i];
    }

    assert(data_size == expected_size);

    int* strides = compute_strides(shape, ndim);

    Tensor t;

    t.data = data;
    t.shape = shape;
    t.strides = strides;
    t.ndim = ndim;

    return t;
}

Tensor zeros(int* shape,int ndim) {

    int size = 1;

    for(int i = 0; i < ndim; i++) {
        size *= shape[i];
    }

    float* data = calloc(size, sizeof(float));

    return tensor_init(data,size,shape,ndim);
}

Tensor matmul(Tensor* a, Tensor* b) {

    // === 2D MATRIX MULTIPLICATION ===
    if (a->ndim == 2 && b->ndim == 2) {

        assert(a->shape[1] == b->shape[0]);

        int m = a->shape[0];
        int n = b->shape[1];
        int k = a->shape[1];

        float* result = calloc(m * n, sizeof(float));

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0.0;
                for (int l = 0; l < k; l++) {
                    sum += a->data[i * k + l] * b->data[l * n + j];
                }
                result[i * n + j] = sum;
            }
        }    
        
        int shape[2] = {m,n};

        return tensor_init(result,m*n,shape,2);
    }

    // === 4D BATCHED MATRIX MULTIPLICATION ===
    if (a->ndim == 4 && b->ndim == 4) {

        int batch = a->shape[0];
        int n_heads = a->shape[1];
        int seq1 = a->shape[2];
        int inner_dim = a->shape[3];
        int seq2 = b->shape[3];

        assert(a->shape[0] == b->shape[0]);
        assert(a->shape[1] == b->shape[1]);
        assert(b->shape[2] == inner_dim);

        int total_size = batch * n_heads * seq1 * seq2;

        float* result = calloc(total_size, sizeof(float));

        for (int batch_idx = 0; batch_idx < batch; batch_idx++) {
            for (int head_idx = 0; head_idx < n_heads; head_idx++) {
                for (int i = 0; i < seq1; i++) {
                    for (int j = 0; j < seq2; j++) {
                        float sum = 0.0;
                        for (int l = 0; l < inner_dim; l++) {
                            int a_idx =((batch_idx * n_heads + head_idx)* seq1 + i)* inner_dim + l;

                            int b_idx = ((batch_idx * n_heads + head_idx) * inner_dim + l) * seq2 + j;

                            sum += a->data[a_idx] * b->data[b_idx];
                        }

                        int out_idx = ((batch_idx * n_heads + head_idx) * seq1 + i) * seq2 + j;

                        result[out_idx] = sum;
                    }
                }
            }
        }

        int shape[4] = {batch,n_heads,seq1,seq2};

        return tensor_init(result,total_size,shape,4);
    }

    assert(0 && "Unsupported matmul shapes");
}




