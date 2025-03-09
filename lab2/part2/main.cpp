#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <fstream>
#include <cmath>
#include <omp.h>

using namespace std;

double func(double x)
{
    return exp(-x * x);
}

double integrate(double (*func)(double), double a, double b, int n)
{
    double h = (b - a) / n;
    double sum = 0.0;

#pragma omp parallel
    {
        int count_thtd = omp_get_num_threads();
        int thrg_id = omp_get_thread_num();
        int items_per_thread = n / count_thtd;
        int st_with = thrg_id * items_per_thread;
        int for_cur_thrd = (thrg_id == count_thtd - 1) ? (n - 1) : (st_with + items_per_thread - 1);
        double sumloc = 0.0;

        for (int i = st_with; i <= for_cur_thrd; i++)
            sumloc += func(a + h * (i + 0.5));
#pragma omp atomic
        sum += sumloc;
    }
    sum *= h;
    return sum;
}

template <typename T>
void write_to_csv(ofstream &file, T *arr)
{
    for (int i = 0; i < 7; i++)
    {
        file << arr[i] << ",";
    }
    file << arr[7] << endl;
}

int main()
{
    ofstream file("res.csv");

    int count_thrd_arr[8] = {1, 2, 4, 7, 8, 16, 20, 40};
    double delta_t_arr[8] = {0};
    double t = 0;
    int a = -4, b = 4, n = 40'000'000;

    write_to_csv(file, count_thrd_arr);

    for (int i = 0; i < 8; ++i)
    {
        omp_set_num_threads(count_thrd_arr[i]);

        t = omp_get_wtime();
        integrate(func, a, b, n);
        delta_t_arr[i] = omp_get_wtime() - t;
    }

    write_to_csv(file, delta_t_arr);

    file.close();
    return 0;
}