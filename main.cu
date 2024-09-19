#include "kernels.h"
#include "operations.h"

template<int N, int M, int K = N, int L = M>
void test_pw_broadcast()
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
    broadcast_kernel<OpPlus><<<grid_size, block_size>>>(a, b, c);

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


template<int N, int M, int K = N, int L = M>
void test_reduce()
{
    auto a_host = allocate_host_tensor<float>(Shape(N, M));
    auto c_host = allocate_host_tensor<float>(Shape(K, L));

    auto a = allocate_device_tensor<float>(a_host.shape);
    auto c = allocate_device_tensor<float>(c_host.shape);

    const int count = size(a);
    for (int i = 0; i < count; ++i)
    {
        a_host.data[i] = i;
    }

    copy_to_device(a_host, a);

    dim3 block_size = 128;
    dim3 grid_size = 2;
    reduce_kernel<OpPlus><<<grid_size, block_size>>>(a, c);

    copy_to_host(c, c_host);

    print_tensor("A", a_host);
    print_tensor("C", c_host);

    free_device_tensor(a);
    free_device_tensor(c);

    free_host_tensor(a_host);
    free_host_tensor(c_host);
}

template<int N, int M, int K>
void test_matmul()
{
    const auto problem = Shape(N, M, K);

    auto a_host = allocate_host_tensor<float>(Shape(N, K));
    auto b_host = allocate_host_tensor<float>(Shape(M, K));
    auto c_host = allocate_host_tensor<float>(Shape(N, M));

    auto a = allocate_device_tensor<float>(Shape(N, 1, K));
    auto b = allocate_device_tensor<float>(Shape(1, M, K));
    auto c = allocate_device_tensor<float>(Shape(N, M, 1));

    for (int i = 0; i < size(a); ++i)
    {
        a_host.data[i] = i;
    }

    for (int i = 0; i < size(b); ++i)
    {
        b_host.data[i] = i % 3;
    }

    copy_to_device(a_host, a);
    copy_to_device(b_host, b);

    dim3 block_size = 128;
    dim3 grid_size = 2;
    contract_kernel<OpMultiply, OpPlus><<<grid_size, block_size>>>(problem, a, b, c);

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
    // Broadcasting
    test_pw_broadcast<3, 4, 3, 1>();
    test_pw_broadcast<3, 4, 1, 4>();

    // Reduction
    test_reduce<3, 4, 3, 1>();
    test_reduce<3, 4, 1, 4>();

    // Matrix Multiplication
    test_matmul<2, 3, 4>();

    return 0;
}