#include <iostream>
#include <vector>
#include <cstdlib>
#include <omp.h>
using namespace std;

void merge(vector<int>& arr, int l, int m, int r) {
    int n1 = m - l + 1;
    int n2 = r - m;

    vector<int> L(n1), R(n2);

    for (int i = 0; i < n1; i++)
        L[i] = arr[l + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[m + 1 + j];

    int i = 0, j = 0, k = l;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

void mergeSort(vector<int>& arr, int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;

        #pragma omp parallel sections
        {
            #pragma omp section
            mergeSort(arr, l, m);
            #pragma omp section
            mergeSort(arr, m + 1, r);
        }

        merge(arr, l, m, r);
    }
}

int main() {
    int n = 10000; 
    vector<int> arr(n);

    for (int i = 0; i < n; i++) {
        arr[i] = rand() % n;
    }

    cout << "\nOriginal vector:\n\n" ;
    for (int num : arr)
        cout << num << " ";
    cout << endl;

    double start, end;
    start = omp_get_wtime();

    mergeSort(arr, 0, n - 1);
    
    end = omp_get_wtime();
    cout<<"\nThe sorted array is: \n\n";
    for (int num : arr)
        cout << num << " ";
    cout << endl;
    cout << "\nParallel Merge Sort: " << end - start<<endl;

    system("pause");
    return 0;
}