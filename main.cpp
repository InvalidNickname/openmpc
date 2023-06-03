#include <iostream>
#include <chrono>
#include <random>
#include "matrix_math.h"

typedef std::chrono::high_resolution_clock Time;
typedef std::chrono::duration<float> FSec;

const int N = 500;
const bool CHECK_ORTHOGONAL = false;
const bool PRINT_Q = false;
const bool PRINT_U = false;
const bool PRINT_B = false;
const bool CHECK_MULTIPLICATION = false;
const bool USE_OMP = true;
#define USE_MASM false

/// Заполнение верхне-треугольной матрицы случайными числами
/// \param mat - матрица n*n для заполнения
/// \param n - размеры матрицы
void fillUpperTriangle(double *mat, int n) {
  std::default_random_engine engine(0);
  std::uniform_real_distribution<float> distribution(-1.0, 1.0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < i; j += 1) {
      mat[i * n + j] = 0.;
    }
    for (int j = i; j < n; j += 1) {
      mat[i * n + j] = distribution(engine);
    }
  }
}

/// Вывод матрицы в консоль
/// \param mat - матрица размерами n*m
void matToConsole(double *mat, int n, int m) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      if (abs(mat[i * m + j]) < 10e-7) {
        std::cout << "0 ";
      } else {
        std::cout << mat[i * m + j] << " ";
      }
    }
    std::cout << "\n";
  }
}

/// Генерация ортогональной симметрической матрицы
/// \param n - размеры матрицы
/// \return - ортогональная симметрическая матрица n*n
double *generateOrthogonalMatrix(int n) {
  std::default_random_engine engine(0);
  std::uniform_real_distribution<float> distribution(-1.0, 1.0);
  auto mat = new double[n];
  for (int i = 0; i < n; ++i) {
    mat[i] = distribution(engine) - distribution(engine);
  }
  auto norm = euclideanNorm(mat, n);
  for (int i = 0; i < n; ++i) {
    mat[i] /= norm;
  }
  mat = dotProduct(transpose(mat, 1, n), mat, n, 1, n);
  matrixMultiply(mat, -2, n, n);
  for (int i = 0; i < n; ++i) {
    mat[i * n + i] += 1;
  }
  return mat;
}

/// Проверка ортогональности матрицы, вывод в консоль A*A.T
/// \param mat - матрица размера n*n
void checkIfOrthogonal(double *mat, int n) {
  matToConsole(dotProduct(mat, transpose(mat, n, n), n, n, n), n, n);
}

extern "C" double multiplyRows(const double *Q, const double *U, int k, int i, int j, int n);

/// Перемножение матриц Q*U*Q.T
/// \param Q - симметрическая ортотогональная матрица размера n*n
/// \param U - верхне-треугольная матрица размера n*n
/// \return - результат перемножения, матрица размера n*n
double *multiply(const double *Q, const double *U, const int n, bool use_omp, bool use_masm) {
  auto *res = new double[n * n];
  #pragma omp parallel for if(use_omp)
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      res[i * n + j] = 0;
      for (int k = 0; k < n; ++k) {
        double temp = 0;
        #if USE_MASM == true
        temp = multiplyRows(&Q[0], &U[0], k, i, j, n);
        #else
        for (int l = k; l < n; ++l) {
          temp += Q[l * n + j] * U[k * n + l];
        }
        temp *= Q[i * n + k];
        #endif
        res[i * n + j] += temp;
      }
    }
  }
  return res;
}

/// Проверка правильности перемножения B=Q*U*Q.T, вывод U и N=Q*B*Q.T, при правильности U==N
/// \param Q - ортогональная симметрическая матрица размера n*n
/// \param U - верхнетреугольная матрица размера n*n
/// \param B - результат перемножения Q*U*Q.T, матрица размера n*n
void checkIfTrue(double *Q, double *U, double *B, int n) {
  auto *check = dotProduct(Q, B, n, n, n);
  check = dotProduct(check, Q, n, n, n);
  matToConsole(check, n, n);
  matToConsole(U, n, n);
}

int main() {
  auto *U = new double[N * N];
  fillUpperTriangle(U, N);

  auto *Q = generateOrthogonalMatrix(N);

  if (PRINT_Q) {
    matToConsole(Q, N, N);
  }

  if (PRINT_U) {
    matToConsole(U, N, N);
  }

  if (CHECK_ORTHOGONAL) {
    checkIfOrthogonal(Q, N);
  }

  auto ts = Time::now();

  auto *A = multiply(Q, U, N, USE_OMP, USE_MASM);

  auto te = Time::now();
  FSec dur = te - ts;
  std::cout << dur.count() * 1000 << "\n\n";

  if (PRINT_B) {
    matToConsole(A, N, N);
  }

  if (CHECK_MULTIPLICATION) {
    checkIfTrue(Q, U, A, N);
  }

  delete[] U;
  delete[] Q;
  delete[] A;
  return 0;
}
