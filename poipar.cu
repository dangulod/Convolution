#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <float.h>

void handleCudaError(cudaError_t cudaERR)
{
    if (cudaERR != cudaSuccess)
    {
        printf("CUDA ERROR : %s\n", cudaGetErrorString(cudaERR));
    }
}

struct paretoParam
{
    float p;
    float xi;
    float psi;
    float u;
    paretoParam() {}
    __device__ paretoParam(float p, float xi, float psi, float u) : p(p), xi(xi), psi(psi), u(u) { }
};

__device__ static float Brent_fmin(float ax, float bx, float (*f)(float, void *), void *info)
{
    const float c = (3. - sqrt(5.)) * .5;

    float a, b, d, e, p, q, r, u, v, w, x;
    float t2, fu, fv, fw, fx, xm, eps, tol, tol1, tol3;

    eps = DBL_EPSILON;
    tol = DBL_EPSILON;
    tol1 = eps + 1.;
    eps = sqrt(eps);

    a = ax;
    b = bx;
    v = a + c * (b - a);
    w = v;
    x = v;

    d = 0.;/* -Wall */
    e = 0.;
    fx = (*f)(x, info);
    fv = fx;
    fw = fx;
    tol3 = tol / 3.;

    for(int i(0); i < 300; i++) {
    // for(;;) {
    xm = (a + b) * .5;
    tol1 = eps * fabs(x) + tol3;
    t2 = tol1 * 2.;


    if (fabs(x - xm) <= t2 - (b - a) * .5) break;
    p = 0.;
    q = 0.;
    r = 0.;
    if (fabs(e) > tol1) {

        r = (x - w) * (fx - fv);
        q = (x - v) * (fx - fw);
        p = (x - v) * q - (x - w) * r;
        q = (q - r) * 2.;
        if (q > 0.) p = -p; else q = -q;
        r = e;
        e = d;
    }

    if (fabs(p) >= fabs(q * .5 * r) ||
        p <= q * (a - x) || p >= q * (b - x)) {

        if (x < xm) e = b - x; else e = a - x;
        d = c * e;
    }
    else {

        d = p / q;
        u = x + d;

        if (u - a < t2 || b - u < t2) {
        d = tol1;
        if (x >= xm) d = -d;
        }
    }

    if (fabs(d) >= tol1)
        u = x + d;
    else if (d > 0.)
        u = x + tol1;
    else
        u = x - tol1;

    fu = (*f)(u, info);

    if (fu <= fx) {
        if (u < x) b = x; else a = x;
        v = w;    w = x;   x = u;
        fv = fw; fw = fx; fx = fu;
    } else {
        if (u < x) a = u; else b = u;
        if (fu <= fw || w == x) {
        v = w; fv = fw;
        w = u; fw = fu;
        } else if (fu <= fv || v == x || v == w) {
        v = u; fv = fu;
        }
    }
    }

    return x;
}

__device__ float dPareto(float x, float xi, float psi, float u)
{
    return 1 - std::pow((1 + xi / psi * (x - u)), (-1 / xi));
}

__device__ float fitness(float x, void *info)
{
    paretoParam *Q = (paretoParam*)info;

    return std::abs(dPareto(x, Q->xi, Q->psi, Q->u) - Q->p);
}

__device__  float qPareto(float p, float xi, float psi, float u)
{
    paretoParam info(p, xi, psi, u);

    float min(Brent_fmin(0, 1e5, fitness, &info));

    return min;
}


__device__ float convolve(curandState *state, float lambda, float xi, float psi, float u)
{
    int freq(curand_poisson(state, lambda));
    float loss(0);

    for (int i = 0; i < freq; i++)
    {
        loss += qPareto(curand_uniform(state), xi, psi, u);
    }

    return loss;
}

__global__ void loss(float *a, float lambda, float xi, float psi, float u, int n)
{
    curandState state;

    int i(threadIdx.x + blockDim.x * blockIdx.x);

    while (i < n)
    {
        curand_init(1234 + i, 0, 0, &state);

        a[i] = convolve(&state, lambda, xi, psi, u);

        i += blockDim.x * gridDim.x;
    }
}

float * poipar(float lambda, float xi, float psi, float u, int n)
{
    float *a, *d_a;
    size_t size(sizeof(float) * n);

    a = new float[size];
    handleCudaError(cudaMalloc(&d_a, size));

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);

    int threads = props.maxThreadsPerBlock;
    int blocks(std::min(0xFFFF, (n + threads - 1) / threads));

    dim3 block(threads, 1, 1);
    dim3 threa(blocks, 1, 1);

    loss<<<block, threa>>>(d_a, lambda, xi, psi, u, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("CUDA ERROR while executing the kernel: %s\n",cudaGetErrorString(err));
    }

    handleCudaError(cudaMemcpy(a, d_a, size, cudaMemcpyDeviceToHost));

    return a;
}
