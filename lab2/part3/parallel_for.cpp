#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>

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

int main()
{
    int num_thrd = 4;
    omp_set_num_threads(num_thrd);

    vector<vector<double>> A;
    int N = 20'000;

    A.resize(N, vector<double>(N, 1.0));
#pragma omp parallel for
    for (int i = 0; i < N; ++i)
    {
        A[i][i] = 2.0;
    }

    vector<double> b(N, N + 1.0);
    vector<double> x(N, 0.0);

    vector<double> r = b;
    vector<double> z = r;

    double eps = 1e-6;
    int max_iter = 1;
    int iter = 0;

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
    cout << "Count threads: " << num_thrd << endl;
    cout << "Count iter: " << iter << endl;
    return 0;
}