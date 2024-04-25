#include <iostream>
#include <stdlib.h>
#include <omp.h>
using namespace std;

void bubble(int *a, int n)
{
    for (int i = 0; i < n; i++)
    {
        int first = i % 2;

        #pragma omp parallel for shared(a, first)
        for (int j = first; j < n - 1; j += 2)
        {
            if (a[j] > a[j + 1])
            {
                swap(a[j], a[j + 1]);
            }
        }
    }
}

void swap(int &a, int &b)
{
    int test = a;
    a = b;
    b = test;
}

int main()
{

    int A[10000];
    int SIZE = 10000;
    for (int i = 0; i < SIZE; i++)
    {
        A[i] = rand() % SIZE;
    }

    double start, end;
    start = omp_get_wtime();
    bubble(A, SIZE);
    end = omp_get_wtime();

    cout << "\n sorted array is=>";
    for (int i = 0; i < SIZE; i++)
    {
        cout << A[i] << " ";
    }

    cout << "\nTime required: " << end - start;

    system("pause");
    return 0;
}