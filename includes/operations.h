#pragma once

#include "constants.h"

struct OpPlus
{
    template <typename ElementA, typename ElementB>
    DEVICE_FUNCTION static auto apply(const ElementA a, const ElementB b)
    {
        return a + b;
    }

    template <typename ElementA, typename ElementB>
    DEVICE_FUNCTION static void atomic(ElementA& a, const ElementB b)
    {
        atomicAdd(&a, b);
    }
};

struct OpMultiply
{
    template <typename ElementA, typename ElementB>
    DEVICE_FUNCTION static auto apply(const ElementA a, const ElementB b)
    {
        return a * b;
    }
};
