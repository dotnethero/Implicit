#pragma once

#include <iostream>
#include <iomanip>
#include <cuda_runtime_api.h>

#include "constants.h"
#include "layout.h"

template<typename Element, typename Shape>
struct Tensor {
    const int rank;
    Shape shape;
    Element* data;

    DEVICE_FUNCTION explicit Tensor(Element* data, const Shape shape): rank(shape.rank), shape(shape), data(data)
    {
    }

    template<typename... Coords>
    DEVICE_FUNCTION Element& operator()(const Coords... coords)
    {
        const auto index = coords_to_index(coords..., shape);
        return data[index];
    }
};

template<typename Element, typename Shape>
DEVICE_FUNCTION int rank(const Tensor<Element, Shape> tensor)
{
    return rank(tensor.shape);
}

template<typename Element, typename Shape>
DEVICE_FUNCTION int size(const Tensor<Element, Shape> tensor)
{
    return size(tensor.shape);
}

// allocation

template<typename Element, typename Shape>
auto allocate_device_tensor(const Shape shape)
{
    Element* data = nullptr;
    auto status = cudaMalloc(&data, size(shape) * sizeof(Element));
    if (status != cudaSuccess)
    {
        throw std::exception("Failed to allocate device memory");
    }
    return Tensor<Element, Shape>(data, shape);
}

template<typename Element, typename Shape>
auto allocate_host_tensor(const Shape shape)
{
    Element* data = nullptr;
    auto status = cudaMallocHost(&data, size(shape) * sizeof(Element));
    if (status != cudaSuccess)
    {
        throw std::exception("Failed to allocate host memory");
    }
    return Tensor<Element, Shape>(data, shape);
}

template<typename Element, typename Shape>
auto free_device_tensor(Tensor<Element, Shape> tensor)
{
    auto status = cudaFree(tensor.data);
    if (status != cudaSuccess)
    {
        throw std::exception("Failed to free device memory");
    }
}

template<typename Element, typename Shape>
auto free_host_tensor(Tensor<Element, Shape> tensor)
{
    auto status = cudaFreeHost(tensor.data);
    if (status != cudaSuccess)
    {
        throw std::exception("Failed to free host memory");
    }
}

// copy

template<typename Element, typename Shape>
auto copy_to_device(const Tensor<Element, Shape> src, Tensor<Element, Shape> dst)
{
    auto status = cudaMemcpy(dst.data, src.data, size(dst) * sizeof(Element), cudaMemcpyHostToDevice);
    if (status != cudaSuccess)
    {
        throw std::exception("Failed to copy data to device");
    }
}

template<typename Element, typename Shape>
auto copy_to_host(const Tensor<Element, Shape> src, Tensor<Element, Shape> dst)
{
    auto status = cudaMemcpy(dst.data, src.data, size(dst) * sizeof(Element), cudaMemcpyDeviceToHost);
    if (status != cudaSuccess)
    {
        throw std::exception("Failed to copy data to device");
    }
}

template<typename Element, typename Shape>
void print_tensor(const char* tag, const Tensor<Element, Shape> tensor)
{
    std::cout << tag << " = " << std::endl;
    const auto count = size(tensor);
    for (int i = 0; i < count; ++i)
    {
        for (int j = 0; j < Shape::Width; ++j)
        {
            if (tensor.shape.strides[j] == 0 ||
                tensor.shape.strides[j] == 1)
                continue;

            auto mod = i % tensor.shape.strides[j];
            auto div = i / tensor.shape.strides[j];
            if (div != 0 && mod == 0)
            {
                std::cout << std::endl;
            }
        }
        std::cout << std::setw(3) << tensor.data[i];
    }
    std::cout << std::endl;
    std::cout << std::endl;
}
