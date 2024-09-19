#include <iostream>
#include "includes/tensor.h"

template<
    typename ElementA,
    typename LayoutA,
    typename ElementB,
    typename LayoutB,
    typename ElementC,
    typename LayoutC>
DEVICE_KERNEL void plus_kernel(
    Tensor<ElementA, LayoutA> a,
    Tensor<ElementB, LayoutB> b,
    Tensor<ElementC, LayoutC> c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size(c))
    {
        auto coord = index_to_coords(i, c.shape);
        c(coord) = a(coord) + b(coord);
    }
}

template<int N, int M, int K = N, int L = M>
void test()
{
    auto a_host = allocate_host_tensor<float>(Shape(N, M));
    auto b_host = allocate_host_tensor<float>(Shape(K, L));
    auto c_host = allocate_host_tensor<float>(Shape(N, M));

    auto a = allocate_device_tensor<float>(a_host.shape);
    auto b = allocate_device_tensor<float>(b_host.shape);
    auto c = allocate_device_tensor<float>(c_host.shape);

    const int count = size(a);
    for (int i = 0; i < count; ++i)
    {
        a_host.data[i] = i;
        b_host.data[i] = count - i;
    }

    copy_to_device(a_host, a);
    copy_to_device(b_host, b);

    dim3 block_size = 128;
    dim3 grid_size = 1;
    plus_kernel<<<grid_size, block_size>>>(a, b, c);

    copy_to_host(a, a_host);
    copy_to_host(b, b_host);
    copy_to_host(c, c_host);

    print_tensor("A", a_host);
    print_tensor("B", b_host);
    print_tensor("C", c_host);

    free_device_tensor(a);
    free_device_tensor(b);
    free_device_tensor(c);

    free_host_tensor(a_host);
    free_host_tensor(b_host);
    free_host_tensor(c_host);
}

int main()
{
    test<3, 4, 3, 1>();
    test<3, 4, 1, 4>();
    return 0;
}