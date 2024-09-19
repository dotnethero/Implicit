#pragma once

#include "constants.h"

template<int MaxRank = 4>
struct Shape
{
    constexpr static int Width = MaxRank;
    const int rank;
    int extents[MaxRank];
    int strides[MaxRank];

    template<typename... Dims>
    DEVICE_FUNCTION explicit Shape(const Dims... dims): rank(sizeof...(dims)), extents{1}, strides{0}
    {
        int temp[] = {static_cast<int>(dims)...};
        strides[rank - 1] = 1;
        extents[rank - 1] = temp[rank - 1];
        for (int i = rank - 2; i >= 0; --i)
        {
            extents[i] = temp[i];
            strides[i] = strides[i + 1] * extents[i + 1];
        }
    }
};

template<typename Shape>
DEVICE_FUNCTION int rank(const Shape shape)
{
    return shape.rank;
}

template<typename Shape>
DEVICE_FUNCTION int size(const Shape shape)
{
    int total = 1;
    for (int i = 0; i < shape.rank; ++i)
    {
        total *= shape.extents[i];
    }
    return total;
}

template<typename Shape>
DEVICE_FUNCTION auto index_to_coords(const int index, const Shape shape)
{
    int leftover = index;
    int coords[Shape::Width];

    #pragma unroll
    for (int i = 0; i < Shape::Width; ++i)
    {
        if (shape.strides[i] != 0)
        {
            coords[i] = leftover / shape.strides[i];
            leftover -= coords[i] * shape.strides[i];
        }
    }

    return coords;
}

template<typename Shape>
DEVICE_FUNCTION int coords_to_index(const int coords[], const Shape shape)
{
    int index = 0;

    #pragma unroll
    for (int i = 0; i < Shape::Width; ++i)
    {
        index += coords[i] * shape.strides[i];
    }

    return index;
}
