#ifndef MATRIX_MATH_CPP
#define MATRIX_MATH_CPP

/// Перемножение матриц C=А*B
/// \param a - матрица A размером n*m
/// \param b - матрица B размером m*p
/// \return матрица С размером n*p
double **dotProduct(double **a, double **b, int n, int m, int p) {
  auto **result = new double *[n];
  #pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    result[i] = new double[p];
    for (int j = 0; j < p; ++j) {
      result[i][j] = 0;
      for (int k = 0; k < m; ++k) {
        result[i][j] += a[i][k] * b[k][j];
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
  #pragma omp parallel for
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
  #pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    a[i] -= b[i];
  }
}

/// Умножение матрицы на скаляр
/// \param a - матрица размером n*m
/// \param b - скаляр
void matrixMultiply(double **a, double b, int n, int m) {
  #pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      a[i][j] *= b;
    }
  }
}

/// Инвертирование матрицы
/// \param a - матрица размером n*m
void invert(double **a, int n, int m) {
  #pragma omp parallel for
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      a[i][j] = -a[i][j];
    }
  }
}

/// Транспонирование матрицы
/// \param mat - матрица А размером n*m
/// \return матрица A.T размером m*n
double **transpose(double **mat, int n, int m) {
  auto res = new double *[m];
  #pragma omp parallel for
  for (int i = 0; i < m; ++i) {
    res[i] = new double[n];
    for (int j = 0; j < n; ++j) {
      res[i][j] = mat[j][i];
    }
  }
  return res;
}

#endif