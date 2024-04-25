#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <omp.h>

using namespace std;

// Function to generate a random graph with n vertices and p probability of edge creation
vector<vector<int>> generateRandomGraph(int n, double p) {
    vector<vector<int>> graph(n, vector<int>(n, 0));
    srand(time(NULL));
    for (int i = 0; i < n; ++i) {
        for (int j = i+1; j < n; ++j) {
            if ((double)rand() / RAND_MAX < p) {
                graph[i][j] = graph[j][i] = 1;
            }
        }
    }
    return graph;
}

// Parallel DFS algorithm
void parallelDFS(const vector<vector<int>>& graph, vector<bool>& visited, int currentNode) {
    cout << "Visiting node " << currentNode << " on thread " << omp_get_thread_num() << endl;
    visited[currentNode] = true;

    #pragma omp parallel for
    for (int i = 0; i < graph.size(); ++i) {
        if (graph[currentNode][i] && !visited[i]) {
            parallelDFS(graph, visited, i);
        }
    }
}

int main() {
    int n = 1000; // Number of vertices
    double p = 0.3; // Probability of edge creation
    vector<vector<int>> graph = generateRandomGraph(n, p);
    vector<bool> visited(n, false);
    int startNode = rand() % n; // Start DFS from a random node

    cout << "Starting DFS from node " << startNode << endl;
    parallelDFS(graph, visited, startNode);
    system("pause");
    return 0;
}
