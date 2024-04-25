#include <iostream>
#include <vector>
#include <queue>
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

// Parallel BFS algorithm
void parallelBFS(const vector<vector<int>>& graph, int startNode) {
    int n = graph.size();
    vector<bool> visited(n, false);
    queue<int> q;
    q.push(startNode);
    visited[startNode] = true;

    while (!q.empty()) {
        #pragma omp parallel
        {
            #pragma omp for
            for (int i = 0; i < n; ++i) {
                if (graph[q.front()][i] && !visited[i]) {
                    #pragma omp critical
                    {
                        cout << "Node " << i << " visited by thread " << omp_get_thread_num() << endl;
                        visited[i] = true;
                        q.push(i);
                    }
                }
            }
        }
        q.pop();
    }
}

int main() {
    int n = 1000; // Number of vertices
    double p = 0.3; // Probability of edge creation
    vector<vector<int>> graph = generateRandomGraph(n, p);
    int startNode = rand() % n; // Start BFS from a random node

    cout << "Starting BFS from node " << startNode << endl;
    double start, end;
    start = omp_get_wtime();
    parallelBFS(graph, startNode);
    end = omp_get_wtime();
    cout<<"Total time: "<<end-start<<endl;
    system("pause");
    return 0;
}
