#ifndef MATRIX_MATH_CPP
#define MATRIX_MATH_CPP

/// Перемножение матриц C=А*B
/// \param a - матрица A размером n*m
/// \param b - матрица B размером m*p
/// \return матрица С размером n*p
double *dotProduct(double *a, double *b, int n, int m, int p) {
  auto *result = new double [n*p];
  #pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < p; ++j) {
      result[i*p+j] = 0;
      for (int k = 0; k < m; ++k) {
        result[i*p+j] += a[i*m+k] * b[k*p+j];
      }
    }
  }
  return result;
}

/// Скалярное произведение векторов c=axb
/// \param a - вектор длиной n
/// \param b - вектор длиной n
/// \return скалярное произведение векторов
double scalarProduct(const double *a, const double *b, int n) {
  double product = 0;
  for (int i = 0; i < n; ++i) {
    product += a[i] * b[i];
  }
  return product;
}

/// Евклидова норма вектора
/// \param a - вектор длиной n
/// \return евклидова норма вектора
double euclideanNorm(const double *a, int n) {
  return sqrt(scalarProduct(a, a, n));
}

/// Вычитание двух векторов a=a-b
/// \param a - вектор длиной n
/// \param b - вектор длиной n
void vectorSubtract(double *a, const double *b, int n) {
  for (int i = 0; i < n; ++i) {
    a[i] -= b[i];
  }
}

/// Умножение матрицы на скаляр
/// \param a - матрица размером n*m
/// \param b - скаляр
void matrixMultiply(double *a, double b, int n, int m) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      a[i*m+j] *= b;
    }
  }
}

/// Транспонирование матрицы
/// \param mat - матрица А размером n*m
/// \return матрица A.T размером m*n
double *transpose(const double *mat, int n, int m) {
  auto res = new double [m*n];
  #pragma omp parallel for
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      res[i*n+j] = mat[j*m+i];
    }
  }
  return res;
}

#endif