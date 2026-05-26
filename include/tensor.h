#ifndef TENSOR_H
#define TENSOR_H

typedef struct {
    float* data;
    const int* shape;
    int* strides;
    int ndim;
} Tensor;

Tensor tensor_init(float* data,int data_size,const int* shape,int ndim);

int* compute_strides(const int* shape,int ndim);

Tensor zeros(int* shape, int ndim);

Tensor matmul(Tensor* a, Tensor* b);

Tensor softmax(Tensor* a, int axis);

#endif