#include <iostream>
#include <vector>
#include <cmath>
#include <omp.h>
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
#pragma omp parallel for reduction(+ : result)
    for (size_t i = 0; i < a.size(); ++i)
    {
        result += a[i] * b[i];
    }
    return result;
}

bool count_err(const vector<double> &x, double eps)
{
    double max_error = 0.0;
#pragma omp parallel for reduction(max : max_error)
    for (size_t i = 0; i < x.size(); ++i)
    {
        max_error = max(max_error, abs(x[i] - 1.0));
    }
    return max_error >= eps;
}

template <typename T>
void write_to_csv(ofstream &file, T *arr, int size)
{
    for (int i = 0; i < size - 1; i++)
    {
        file << arr[i] << ",";
    }
    file << arr[size - 1] << endl;
}

int main()
{
    ofstream file("res.csv");

    int count_thrd_arr[COUNT_THRD] = {1, 2, 4, 8, 16, 32, 40};
    double delta_t_arr[COUNT_THRD] = {0};

    write_to_csv(file, count_thrd_arr, COUNT_THRD);

    int N = 50000;
    vector<vector<double>> A(N, vector<double>(N, 1.0));
    vector<double> b(N, N + 1.0);
    vector<double> r = b;
    vector<double> z = r;
    vector<double> r_new(N);
    double eps = 1e-6;
    int iter = 0;

    double alpha = 0.0, num = 0.0, denom = 0.0, beta = 0.0;

    for (int i = 0; i < N; ++i)
    {
        A[i][i] = 2.0;
    }

    for (int t_idx = 0; t_idx < COUNT_THRD; ++t_idx)
    {

        vector<double> x(N, 0.0);

        omp_set_num_threads(count_thrd_arr[t_idx]);

        double t = omp_get_wtime();

#pragma omp parallel
        {
            while (count_err(x, eps))
            {
                vector<double> Az(N);
                alpha = 0.0, num = 0.0, denom = 0.0, beta = 0.0;

#pragma omp parallel for
                for (int i = 0; i < z.size(); ++i)
                {
                    double sum = 0.0;
                    for (int j = 0; j < z.size(); ++j)
                    {
                        sum += A[i][j] * z[j];
                    }
                    Az[i] = sum;
                }

#pragma omp parallel for reduction(+ : num)
                for (size_t i = 0; i < r.size(); ++i)
                {
                    num += r[i] * r[i];
                }

#pragma omp parallel for reduction(+ : denom)
                for (size_t i = 0; i < z.size(); ++i)
                {
                    denom += z[i] * Az[i];
                }

#pragma omp single
                {
                    alpha = num / denom;
                    num = 0.0;
                    denom = 0.0;
                }

#pragma omp for
                for (int i = 0; i < N; ++i)
                {
                    x[i] += alpha * z[i];
                }

#pragma omp for
                for (int i = 0; i < N; ++i)
                {
                    r_new[i] = r[i] - alpha * Az[i];
                }

#pragma omp parallel for reduction(+ : num)
                for (size_t i = 0; i < r_new.size(); ++i)
                {
                    num += r_new[i] * r_new[i];
                }

#pragma omp parallel for reduction(+ : denom)
                for (size_t i = 0; i < r.size(); ++i)
                {
                    denom += r[i] * Az[i];
                }

#pragma omp single
                {
                    beta = num / denom;
                    num = 0.0;
                    denom = 0.0;
                }

#pragma omp for
                for (int i = 0; i < N; ++i)
                {
                    z[i] = r_new[i] + beta * z[i];
                }
            }

            r = r_new;
            ++iter;
        }

        delta_t_arr[t_idx] = omp_get_wtime() - t;

        cout << "Threads: " << count_thrd_arr[t_idx] << " Time: " << delta_t_arr[t_idx] << "s" << endl;
        iter = 0;
    }

    write_to_csv(file, delta_t_arr, COUNT_THRD);
    file.close();

    return 0;
}
