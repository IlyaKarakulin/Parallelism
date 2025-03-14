#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
#include <ctime>
#include <fstream>

#define COUNT_THRD 7

using namespace std;

vector<double> matrix_mult(vector<vector<double>> &A, const vector<double> &v)
{
    int n = v.size();
    vector<double> result(n, 0.0);

#pragma omp parallel for
    for (int i = 0; i < n; ++i)
    {
        double sum = 0.0;
        for (int j = 0; j < n; ++j)
        {
            sum += A[i][j] * v[j];
        }
        result[i] = sum;
    }
    return result;
}

double dot_product(const vector<double> &a, const vector<double> &b)
{
    double result = 0.0;
#pragma omp parallel for
    for (size_t i = 0; i < a.size(); ++i)
    {
        result += a[i] * b[i];
    }
    return result;
}

bool count_err(const vector<double> &x, double eps)
{
    double max_error = 0.0;
#pragma omp parallel for
    for (size_t i = 0; i < x.size(); ++i)
    {
        max_error = max(max_error, abs(x[i] - 1.0));
    }
    return max_error >= eps;
}

template <typename T>
void write_to_csv(ofstream &file, T *arr)
{
    for (int i = 0; i < COUNT_THRD; i++)
    {
        file << arr[i] << ",";
    }
    file << arr[COUNT_THRD - 1] << endl;
}

int main()
{
    ofstream file("res.csv");

    int count_thrd_arr[8] = {1, 2, 4, 8, 16, 32, 40};
    // int count_thrd_arr[COUNT_THRD] = {1};
    double delta_t_arr[COUNT_THRD] = {0};

    write_to_csv(file, count_thrd_arr);

    vector<vector<double>> A;
    int N = 100'000;

    A.resize(N, vector<double>(N, 1.0));
    // #pragma omp parallel for
    for (int i = 0; i < N; ++i)
    {
        A[i][i] = 2.0;
    }

    cout << "!" << endl;

    for (int i = 0; i < COUNT_THRD; ++i)
    {
        double eps = 1e-6;
        double t = 0.0;
        int max_iter = 1;
        int iter = 0;

        vector<double> b(N, N + 1.0);
        vector<double> x(N, 0.0);
        vector<double> r = b;
        vector<double> z = r;

        omp_set_num_threads(count_thrd_arr[i]);

        t = omp_get_wtime();

        while (count_err(x, eps))
        {
            vector<double> Az = matrix_mult(A, z);

            double alpha = dot_product(r, r) / dot_product(z, Az);

#pragma omp parallel for
            for (int i = 0; i < N; ++i)
            {
                x[i] += alpha * z[i];
            }

            vector<double> r_new = r;
#pragma omp parallel for
            for (int i = 0; i < N; ++i)
            {
                r_new[i] -= alpha * Az[i];
            }

            double beta = dot_product(r_new, r_new) / dot_product(r, r);

#pragma omp parallel for
            for (int i = 0; i < N; ++i)
            {
                z[i] = r_new[i] + beta * z[i];
            }

            r = r_new;
            ++iter;
        }

        delta_t_arr[i] = omp_get_wtime() - t;

        cout << delta_t_arr[i] << endl;
    }

    write_to_csv(file, delta_t_arr);
    file.close();

    return 0;
}