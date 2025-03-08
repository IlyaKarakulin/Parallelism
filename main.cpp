#include <iostream>
#include <vector>
#include <cmath>

#ifdef USE_DOUBLE
#define ARR_TYPE double
#define TYPE_NAME "double"
#else
#define ARR_TYPE float
#define TYPE_NAME "float"
#endif

constexpr size_t SIZE = 10'000'000;
constexpr double PI2 = 6.2831853071795864769252;

int main()
{
    std::cout << "Using data type: " << TYPE_NAME << std::endl;

    using user_type = ARR_TYPE;
    std::vector<user_type> arr(SIZE);
    user_type sum = 0;

    for (size_t i = 0; i < SIZE; ++i)
    {
        arr[i] = std::sin(PI2 * i / SIZE);
        sum += arr[i];
    }

    std::cout << sum << std::endl;

    return 0;
}
