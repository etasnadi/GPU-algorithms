#include <cuda_fp16.h>

#include <cmath>
#include <iomanip>
#include <iostream>

int main()
{
    // Change x and repeat the experiment around 0.0, 1.0, 2048.0, and 65000.0.
    // For normal numbers (s, f, e): x = (-1)**s x 1.f x 2**e-15
    // Example: 1.0 = (-1)*0 * 1.0 * 2**15-15
    //  s = 0
    //  f = 0
    //  e = 15
    float x = 1.64f;

    // Convert the float number to the nearest half
    __half h = __float2half_rn(x);
    unsigned short bits = __half_as_ushort(h);
    float reconstructed = __half2float(h);

    float signed_error = reconstructed - x;
    float absolute_error = std::abs(signed_error);
    float relative_error = 0.0f;
    if (x != 0.0f) {
        relative_error = absolute_error / std::abs(x);
    }

    std::cout << std::fixed << std::setprecision(10);
    std::cout << "Original:       " << x << '\n';
    std::cout << "FP16 fields:    sign | exponent | significand\n";
    std::cout << "FP16 bits:      " << ((bits >> 15) & 1) << " | ";
    for (int bit = 14; bit >= 10; --bit) {
        std::cout << ((bits >> bit) & 1);
    }
    std::cout << " | ";
    for (int bit = 9; bit >= 0; --bit) {
        std::cout << ((bits >> bit) & 1);
    }
    std::cout << '\n';
    std::cout << "Reconstructed:  " << reconstructed << '\n';
    std::cout << "Signed error:   " << signed_error << '\n';
    std::cout << "Absolute error: " << absolute_error << '\n';
    std::cout << "Relative error: " << relative_error << '\n';

    return 0;
}
