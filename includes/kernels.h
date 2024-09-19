#pragma once

#include "constants.h"
#include "tensor.h"

template<
    typename Operation,
    typename ElementA, typename LayoutA,
    typename ElementB, typename LayoutB,
    typename ElementC, typename LayoutC>
DEVICE_KERNEL void broadcast_kernel(
    const Tensor<ElementA, LayoutA> a,
    const Tensor<ElementB, LayoutB> b,
          Tensor<ElementC, LayoutC> c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size(c))
    {
        auto coord = index_to_coords(i, c.shape);
        c(coord) = Operation::apply(a(coord), b(coord));
    }
}

template<
    typename Operation,
    typename ElementA, typename LayoutA,
    typename ElementC, typename LayoutC>
DEVICE_KERNEL void reduce_kernel(
    const Tensor<ElementA, LayoutA> a,
          Tensor<ElementC, LayoutC> c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size(a))
    {
        auto coord = index_to_coords(i, a.shape);
        Operation::atomic(c(coord), a(coord));
    }
}

template<
    typename BitwiseOp,
    typename ReduceOp,
    typename Problem,
    typename ElementA, typename LayoutA,
    typename ElementB, typename LayoutB,
    typename ElementC, typename LayoutC>
DEVICE_KERNEL void contract_kernel(
    Problem problem,
    const Tensor<ElementA, LayoutA> a,
    const Tensor<ElementB, LayoutB> b,
          Tensor<ElementC, LayoutC> c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size(problem))
    {
        const auto coord = index_to_coords(i, problem);
        const auto product = BitwiseOp::apply(a(coord), b(coord));
        ReduceOp::atomic(c(coord), product);
    }
}
