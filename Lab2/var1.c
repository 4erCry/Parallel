#include <iostream> 
#include <cmath> 
#include <sys/time.h> 
#include <cstdlib> 
#include <cstring> 
#include <stdio.h>
#include <omp.h> 

#define e 1e-10 

void matrixColumnMul(double *matrix, double *column, int N, double *res_column)
{
#pragma omp parallel for schedule(static, (int)N/omp_get_num_threads()) private(j)
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++) 
		{
			res_column[i] += matrix[i*N + j] * column[j];
		}
	}
}

void columnSub(double *column_minuend, double *column_subtrahend, int N, double *res_column) 
{
#pragma omp parallel for private(i)
	for (int i = 0; i < N; i++)
	{
		res_column[i] = column_minuend[i] - column_subtrahend[i];
	}
}

void columnNumMul(double *column, double num, int N, double *res_column) 
{
#pragma omp parallel for private(i)
	for (int i = 0; i < N; i++) 
	{
		res_column[i] = column[i] * num;
	}
}

void debuf(double *cleanable, int N) 
{
	for (int i = 0; i < N; i++) 
	{
		cleanable[i] = 0;
	}
}

bool endChecker(double *matrix, double *xn, double *b_column, int N, int count)
{
	double first_part, second_part;
	double *buf = (double*)calloc(N, sizeof(double));
	double *buf2 = (double*)calloc(N, sizeof(double));
	matrixColumnMul(matrix, xn, N, buf);
	columnSub(buf, b_column, N, buf2);
#pragma omp parallel for reduction(+:first_part)
	for (int i = 0; i < N; i++)
	{
		first_part += buf2[i] * buf2[i];
	}
#pragma omp parallel for reduction(+:second_part)
	for (i = 0; i < N; i++)
	{
		second_part += b_column[i] * b_column[i];
	}
	first_part = sqrt(first_part);
	second_part = sqrt(second_part);
	double final = first_part / second_part;
	if (count > 10) final *= 0.01;
	debuf(buf, N);
	debuf(buf2, N);
	if (final < e)
	{
		return true;
	}
	else 
	{
		return false;
	}
}

void matrixFiller(double *matrix, int N) 
{
#pragma omp parallel for private(i)
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++) 
		{
			if (i != j) 
			{
				matrix[i * N + j] = 1.0f;
			}
			else 
			{
				matrix[i * N + j] = 2.0f;
			}
		}
	}
}

void b_columnCreator(double* column, double* matrix, int N)
{
	double *u = (double*)calloc(N, sizeof(double));
#pragma omp parallel for private(i)
	for (int i = 0; i < N; i++)
	{
		u[i] = sin((2 * M_PI*i) / N);
	}
	matrixColumnMul(matrix, u, N, column);
}

void b_columnFiller(double* column, int N) 
{
	for (int i = 0; i < N; i++) {
		column[i] = N + 1;
	}
}

double taoCalculate(int N, double* y, double* matrix)
{
	double first_part = 0, second_part = 0;
	double *temp = (double*)calloc(N, sizeof(double));
	matrixColumnMul(matrix, y, N, temp);
#pragma omp parallel for reduction(+:first_part)
	for (int i = 0; i < N; i++) 
	{
		first_part += y[i] * temp[i];
	}
#pragma omp parallel for reduction(+:second_part)
	for (i = 0; i < N; i++)
	{
		second_part += temp[i] * temp[i];
	}

	debuf(temp, N);
	return first_part / second_part;
}

int main() 
{
	int N = 17000;
	omp_set_num_threads(1);
	double t;
	bool flag = false;
	int count = 0;
	struct timeval start, end;
	double *matrix = (double*)calloc((N*N), sizeof(double));
	double *xn = (double*)calloc(N, sizeof(double));
	double *xn1 = (double*)calloc(N, sizeof(double));
	double *buffer = (double*)calloc(N, sizeof(double));
	double *b_column = (double*)calloc(N, sizeof(double));
	double *y = (double*)calloc(N, sizeof(double));
	double *ty = (double*)calloc(N, sizeof(double));
	matrixFiller(matrix, N);
	int imode = 1;
	if (imode)
	{
		b_columnCreator(b_column, matrix, N);
	}
	else
	{
		b_columnFiller(b_column, N);
	}
	gettimeofday(&start, NULL);
	while (!flag) 
	{
		count++;
		debuf(buffer, N);
		matrixColumnMul(matrix, xn, N, buffer);
		columnSub(buffer, b_column, N, y);
		t = taoCalculate(N, y, matrix);
		columnNumMul(y, t, N, ty);
		columnSub(xn, ty, N, xn1);
		flag = endChecker(matrix, xn1, b_column, N, count);
		std::memcpy(xn, xn1, (N * sizeof(double)));
	}
	gettimeofday(&end, NULL);
	double dt_sec = (end.tv_sec - start.tv_sec);
	double dt_usec = (end.tv_usec - start.tv_usec);
	double dt = dt_sec + 0.000001 * dt_usec;
	printf("time %lf \n", dt);
	return 0;
}
