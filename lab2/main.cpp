#include <iostream>
#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <fstream>
#include <omp.h>

using namespace std;

void init(double **matrix, double *vector, int n)
{
#pragma omp parallel
    {
        int cnt_thrd = omp_get_num_threads();
        int thrd_id = omp_get_thread_num();

        int cnt_matrix_str_for_thrd = n / cnt_thrd;
        int st_with = thrd_id * cnt_matrix_str_for_thrd;
        int last_str = st_with + cnt_matrix_str_for_thrd;

        for (int i = st_with; i < last_str; i++)
        {
            for (int j = 0; j < n; j++)
                matrix[i][j] = i + j;
        }

        for (int j = st_with; j < last_str; j++)
            vector[j] = j;
    }
}

void matrix_product(double **matrix, double *vector, double *res_vector, int n)
{
#pragma omp parallel
    {
        int cnt_thrd = omp_get_num_threads();
        int thrd_id = omp_get_thread_num();

        int cnt_matrix_str_for_thrd = n / cnt_thrd;
        int st_with = thrd_id * cnt_matrix_str_for_thrd;
        int last_str = st_with + cnt_matrix_str_for_thrd;

        if (thrd_id == cnt_thrd)
            last_str = n;

        for (int i = st_with; i < last_str; ++i)
        {
            res_vector[i] = 0.0;
            for (int j = 0; j < n; j++)
                res_vector[i] += matrix[i][j] * vector[j];
        }
    }
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
    double **matrix = nullptr;
    double *vector = nullptr, *res_vector = nullptr;

    write_to_csv(file, count_thrd_arr);

    for (int n = 20000; n <= 40000; n *= 2)
    {
        matrix = new double *[n];
        for (int i = 0; i < n; i++)
            matrix[i] = new double[n];

        vector = new double[n];
        res_vector = new double[n];

        omp_set_num_threads(20);
        init(matrix, vector, n);

        for (int i = 0; i < 8; ++i)
        {
            omp_set_num_threads(count_thrd_arr[i]);

            t = omp_get_wtime();
            matrix_product(matrix, vector, res_vector, n);
            delta_t_arr[i] = omp_get_wtime() - t;
        }

        write_to_csv(file, delta_t_arr);

        for (int i = 0; i < n; ++i)
            free(matrix[n]);
        free(matrix);
        free(vector);
    }

    file.close();
    return 0;
}