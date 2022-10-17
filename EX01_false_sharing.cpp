


#include <immintrin.h>
#include <iostream>
#include <stdint.h>
#include <omp.h>
#include <vector>
#include <chrono>
#include <assert.h>



float X1=1.4142135623730950488;
 float x2=1.7320508075688772935;


#if _WIN32
#include <intrin.h>
uint64_t rdtsc()  // win
    {
    return __rdtsc();
    }

#else

uint64_t rdtsc() // linux
{
    unsigned int lo, hi;
    __asm__ volatile ("rdtsc" : "=a" (lo), "=d" (hi));
    return ((uint64_t)hi << 32) | lo;
}

#endif

#include <iostream>
#include <vector>
#include <chrono>
#include <cstring>
#include "omp.h"

#define NREPEAT 128
#define NTHREADMAX 8
#define STRIDE 1

int main()
{
    int N = 10000000;
    float sum[NTHREADMAX * STRIDE] __attribute__((aligned(64))) = {0};
    std::vector<float> vec(N);
    vec[0] = 0;
    for (int i = 1; i < N; i++) {
        vec[i] = 1;
    }


    auto start = std::chrono::high_resolution_clock::now();
    for (int repeat = 0; repeat < NREPEAT; repeat++) {
        for (int i = 0; i < NTHREADMAX; i++) { sum[i * STRIDE] = 0.0; }

        omp_set_num_threads(8);

#pragma omp parallel
        {

            int thid = omp_get_thread_num();

#pragma omp for
            for (int i = 0; i < N; i++) {
                sum[thid * STRIDE] += vec[i];
            }
        }
    }


    std::chrono::duration<double> time = std::chrono::high_resolution_clock::now() - start;
    std::cout << "Time: " << time.count() << "s\n";

    float sumFinal = 0.0;
    for (int i = 0; i < NTHREADMAX; i++) { sumFinal += sum[i]; }
    printf("sum = %f", sumFinal);

    return 0;
}
