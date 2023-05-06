#include <iostream>
#include <chrono>
#include <random>
#include "matrix_math.h"

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::duration<float> FSec;

void fillUpperTriangle(double **mat, int n) {
  for (int i = 0; i < n; ++i) {
    mat[i] = new double[n];
    #pragma omp parallel shared(mat)
    {
      #pragma omp for
      for (int j = 0; j < i; j += 1) {
        mat[i][j] = 0.;
      }
      #pragma omp for
      for (int j = i; j < n; j += 1) {
        mat[i][j] = 1.;
      }
    }
  }
}

void matToConsole(double **mat, int n, int m) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      if (abs(mat[i][j]) < 10e-7) {
        std::cout << "0 ";
      } else {
        std::cout << mat[i][j] << " ";
      }
    }
    std::cout << "\n";
  }
}

double **generateOrthogonalMatrix(int n) {
  std::default_random_engine engine(0);
  std::uniform_real_distribution<float> distribution(-1.0, 1.0);
  auto mat = new double *[2]{new double[n], new double[n]};
  #pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    mat[0][i] = distribution(engine) - distribution(engine);
  }
  auto norm = euclideanNorm(mat[0], n);
  #pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    mat[0][i] /= norm;
  }
  mat = dotProduct(transpose(mat, 1, n), mat, n, 1, n);
  matrixMultiply(mat, 2, n, n);
  invert(mat, n, n);
  for (int i = 0; i < n; ++i) {
    mat[i][i] += 1;
  }
  return mat;
}

void checkIfOrthogonal(double **mat, int n) {
  matToConsole(dotProduct(mat, transpose(mat, n, n), n, n, n), n, n);
}

double** multiply(double **Q, double **U, int n) {
  auto** res = new double*[n];
  #pragma omp parallel for shared(res)
  for (int i = 0; i < n; ++i) {
    res[i] = new double[n];
    for (int j = 0; j < n; ++j) {
      res[i][j] = 0;
      for (int k = 0; k < n; ++k) {
        for (int l = k; l < n; ++l) {
          res[i][j] += Q[i][k] * Q[l][j];
        }
      }
    }
  }
  return res;
}

int main() {
  int n = 500;

  auto ts = Time::now();

  auto **U = new double *[n];
  fillUpperTriangle(U, n);

  auto **Q = generateOrthogonalMatrix(n);

  //matToConsole(U, n, n);
  //matToConsole(Q, n, n);

  //checkIfOrthogonal(Q, n);

  auto ** A = multiply(Q, U, n);

  auto te = Time::now();
  FSec dur = te - ts;
  std::cout << dur.count() * 1000 << "\n\n";

  //matToConsole(A, n, n);

  for (int i = 0; i < n; ++i) {
    delete[] U[i];
    delete[] Q[i];
  }
  delete[] U;
  delete[] Q;
  return 0;
}
