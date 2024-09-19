#pragma once

#include "constants.h"
#include "tensor.h"

template<
    typename ElementA,
    typename LayoutA,
    typename ElementB,
    typename LayoutB,
    typename ElementC,
    typename LayoutC>
DEVICE_KERNEL void plus_kernel(
    const Tensor<ElementA, LayoutA> a,
    const Tensor<ElementB, LayoutB> b,
          Tensor<ElementC, LayoutC> c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size(c))
    {
        auto coord = index_to_coords(i, c.shape);
        c(coord) = a(coord) + b(coord);
    }
}

template<
    typename ElementA,
    typename LayoutA,
    typename ElementC,
    typename LayoutC>
DEVICE_KERNEL void reduce_plus_kernel(
    const Tensor<ElementA, LayoutA> a,
          Tensor<ElementC, LayoutC> c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size(a))
    {
        auto coord = index_to_coords(i, a.shape);
        atomicAdd(&c(coord), a(coord));
    }
}

template<
    typename Problem,
    typename ElementA,
    typename LayoutA,
    typename ElementB,
    typename LayoutB,
    typename ElementC,
    typename LayoutC>
DEVICE_KERNEL void matmul_kernel(
    Problem problem,
    const Tensor<ElementA, LayoutA> a,
    const Tensor<ElementB, LayoutB> b,
          Tensor<ElementC, LayoutC> c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size(problem))
    {
        const auto coord = index_to_coords(i, problem);
        const auto product = a(coord) * b(coord);
        atomicAdd(&c(coord), product);
    }
}
