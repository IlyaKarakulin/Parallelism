#include <iostream>
#include <vector>
#include <cmath>

using namespace std;

vector<double> matrix_mult(vector<vector<double>> &A, const vector<double> &v)
{
    int n = v.size();
    vector<double> result(n, 0.0);
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < n; ++j)
        {
            result[i] += A[i][j] * v[j];
        }
    }
    return result;
}

double dot_product(const vector<double> &a, const vector<double> &b)
{
    double result = 0.0;
    for (size_t i = 0; i < a.size(); ++i)
    {
        result += a[i] * b[i];
    }
    return result;
}

bool count_err(vector<double> x, double eps)
{
    double max_error = 0.0;
    for (double xi : x)
    {
        max_error = max(max_error, abs(xi - 1.0));
    }

    return max_error >= eps ? 1 : 0;
}

int main()
{
    vector<vector<double>> A;
    int N = 10'000;

    A.resize(N, vector<double>(N, 1.0));
    for (int i = 0; i < N; ++i)
    {
        A[i][i] = 2.0;
    }

    vector<double> b(N, N + 1.0);
    vector<double> x(N, 0.0);

    vector<double> r = b; // r = b - A*x (x = 0)
    vector<double> z = r;

    double eps = 1e-6;
    int max_iter = 1;
    int iter = 0;

    while (count_err(x, eps))
    {
        vector<double> Az = matrix_mult(A, z);

        double alpha = dot_product(r, r) / dot_product(z, Az);

        for (int i = 0; i < N; ++i)
        {
            x[i] += alpha * z[i];
        }

        vector<double> r_new = r;
        for (int i = 0; i < N; ++i)
        {
            r_new[i] -= alpha * Az[i];
        }

        double beta = dot_product(r_new, r_new) / dot_product(r, r);
        for (int i = 0; i < N; ++i)
        {
            z[i] = r_new[i] + beta * z[i];
        }

        r = r_new;
        ++iter;
    }

    // cout << "Res:\n";
    // for (double xi : x)
    // {
    //     printf("%.6f ", xi);
    // }
    cout << "\nCount iter: " << iter << endl;

    return 0;
}