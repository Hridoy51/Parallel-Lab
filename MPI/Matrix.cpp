// #include <bits/stdc++.h>
#include <iostream>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
using namespace std;

int numOfMatrix;
int rowA ,colA , rowB, colB ;

// print a matrix

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  double startTime, endTime;
  if (rank == 0) {
    cout << "Enter the number of matrices :";
    cin >> numOfMatrix;

    cout << "Enter number of row of matrices A :";
    cin >> rowA;
    cout << "Enter number of col of matrices A :";
    cin >> colA;
    cout << "Enter number of row of matrices B :";
    cin >> rowB;
    cout << "Enter number of col of matrices B :";
    cin >> colB;
  }
  MPI_Bcast(&numOfMatrix, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&rowA, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&colA, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&rowB, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&colB, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if (numOfMatrix % size != 0) {
    cout << "Number of matrices should be divisible by number of Process";
    MPI_Finalize();
    return 1;
  }
  int root = 0;
  int perProcess = numOfMatrix / size;
  int matricesA[numOfMatrix][rowA][colA];
  int matricesB[numOfMatrix][rowB][colB];
  int result[perProcess][rowA][colB];
  // initialize matrices in root
  if (rank == root) {
    for (int k = 0; k < numOfMatrix; k++) {
      for (int i = 0; i < rowA; i++) {
        for (int j = 0; j < colA; j++) {
          matricesA[k][i][j] = rand() % 10;
        }
      }
      for (int i = 0; i < rowB; i++) {
        for (int j = 0; j < colB; j++) {
          matricesB[k][i][j] = rand() % 10;
        }
      }
    }
  }

  // starting parrallelly...
  MPI_Barrier(MPI_COMM_WORLD);
  startTime = MPI_Wtime();

  int localA[perProcess][rowA][colA];
  int localB[perProcess][rowB][colB];

  // distribute matrices to the processes
  MPI_Scatter(matricesA, perProcess * rowA * colA, MPI_INT, localA,
              perProcess * rowA * colA, MPI_INT, root, MPI_COMM_WORLD);

  MPI_Scatter(matricesB, perProcess * rowB * colB, MPI_INT, localB,
              perProcess * rowB * colB, MPI_INT, root, MPI_COMM_WORLD);

  // now perform multliplication

  for (int k = 0; k < perProcess; k++) {
    for (int i = 0; i < rowA; i++) {
      for (int j = 0; j < colB; j++) {
        result[k][i][j] = 0;
        for (int l = 0; l < colA; l++) {
          result[k][i][j] += localA[k][i][l] * localB[k][l][j];
        }
      }
    }
  }
  // synchronize all process to record time
  MPI_Barrier(MPI_COMM_WORLD);
  endTime = MPI_Wtime();
  printf("Process %d: Time taken = %f seconds\n", rank, endTime - startTime);

  // gather result from the processes
  int finalresult[numOfMatrix][rowA][colB];
  MPI_Gather(result, perProcess * rowA * colB, MPI_INT, finalresult,
             perProcess * rowA * colB, MPI_INT, root, MPI_COMM_WORLD);
  // print results from root processes
  if (rank == root) {
    cout << "The Result is : ";
    for (int k = 0; k < numOfMatrix; k++) {
      cout << "Result matrix " << k << " :\n";
      for (int i = 0; i < rowA; i++) {
        for (int j = 0; j < colB; j++) {
          cout << finalresult[k][i][j] << " ";
        }
        cout << endl;
      }
    }
  }

  MPI_Finalize();
  return 0;
}
