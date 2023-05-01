#include <iostream>
#include <chrono>
#include <random>

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

double scalarProduct(const double *a, const double *b, int n) {
  double product = 0;
  for (int i = 0; i < n; ++i) {
    product += a[i] * b[i];
  }
  return product;
}

double *scalarProduct(const double *a, const double b, int n) {
  auto *result = new double[n];
  for (int i = 0; i < n; ++i) {
    result[i] = a[i] * b;
  }
  return result;
}

void subtractFromVector(double *a, const double *b, int n) {
  for (int i = 0; i < n; ++i) {
    a[i] -= b[i];
  }
}

double euclideanNorm(const double *a, int n) {
  return sqrt(scalarProduct(a, a, n));
}

void checkIfOrthogonal(double **mat, int n) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < i; ++j) {
      std::cout << scalarProduct(mat[i], mat[j], n) << " ";
    }
    std::cout << euclideanNorm(mat[i], n) << "\n";
  }
}

void matToConsole(double **mat, int n) {
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      std::cout << mat[i][j] << " ";
    }
    std::cout << "\n";
  }
}

void generateOrthogonalMatrix(double **mat, int n) {
  std::default_random_engine engine(0);
  std::uniform_real_distribution<float> distribution(-1.0, 1.0);
  for (int i = 0; i < n; i += 1) {
    mat[i] = new double[n];
    for (int j = 0; j < n; j += 1) {
      mat[i][j] = distribution(engine);
    }
  }
  for (int k = 0; k < n; k += 1) {
    auto *r = new double[n];
    for (int i = 0; i < k; i += 1) {
      r[i] = scalarProduct(mat[k], mat[i], n);
      subtractFromVector(mat[k], scalarProduct(mat[i], r[i], n), n);
    }
    r[k] = euclideanNorm(mat[k], n);
    if (r[k] < 10e-7) {
      throw std::exception("v_i are dependent, exiting");
    }
    mat[k] = scalarProduct(mat[k], 1 / r[k], n);
  }
}

int main() {
  int n = 3;

  auto **U = new double *[n];
  fillUpperTriangle(U, n);

  auto **Q = new double *[n];
  generateOrthogonalMatrix(Q, n);

  //matToConsole(U, n);
  matToConsole(Q, n);

  checkIfOrthogonal(Q, n);

  for (int i = 0; i < n; ++i) {
    delete[] U[i];
    delete[] Q[i];
  }
  delete[] U;
  delete[] Q;
  return 0;
}
