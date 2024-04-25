#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <omp.h>

using namespace std;

// Function to generate a random vector of given size
vector<int> generateRandomVector(int size) {
    vector<int> vec(size);
    srand(time(NULL));
    for (int i = 0; i < size; ++i) {
        vec[i] = rand() % 100; // Generate random numbers between 0 and 99
    }
    return vec;
}

// Parallel reduction for finding the minimum value
int parallelMin(const vector<int>& vec) {
    int minVal = vec[0];
    #pragma omp parallel for reduction(min:minVal)
    for (int i = 0; i < vec.size(); ++i) {
        if (vec[i] < minVal) {
            minVal = vec[i];
        }
    }
    return minVal;
}

// Parallel reduction for finding the maximum value
int parallelMax(const vector<int>& vec) {
    int maxVal = vec[0];
    #pragma omp parallel for reduction(max:maxVal)
    for (int i = 0; i < vec.size(); ++i) {
        if (vec[i] > maxVal) {
            maxVal = vec[i];
        }
    }
    return maxVal;
}

// Parallel reduction for finding the sum
int parallelSum(const vector<int>& vec) {
    int sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < vec.size(); ++i) {
        sum += vec[i];
    }
    return sum;
}

// Parallel reduction for finding the average
double parallelAverage(const vector<int>& vec) {
    int sum = parallelSum(vec);
    return static_cast<double>(sum) / vec.size();
}

int main() {
    int size = 1000; // Size of the random vector
    vector<int> vec = generateRandomVector(size);
    cout<<"The array is: \n"; 
    for (int num : vec)
        cout << num << " ";
    cout << endl;
    // Compute and print min, max, sum, and average in parallel
    cout << "Minimum: " << parallelMin(vec) << endl;
    cout << "Maximum: " << parallelMax(vec) << endl;
    cout << "Sum: " << parallelSum(vec) << endl;
    cout << "Average: " << parallelAverage(vec) << endl;
    system("pause");
    return 0;
}
