#include <iostream>
#include <cmath>
#include <sys/time.h>
#include <cstdlib>
#include <cstring>
#include <stdio.h>
#include <mpi.h>

#define e 1e-9  

void matrixColumnMul(double *matrix, double *column, int N, double *res_column) 
{
	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++) 
		{
			res_column[i] += matrix[i*N + j] * column[j];
		}
	}
}

void debuf(double *cleanable, int N) 
{
	for (int i = 0; i < N; i++) 
	{
		cleanable[i] = 0;
	}
}

void matrixFiller(double *matrix, int N) 
{
	int i, j;

	for (i = 0; i < N; i++) 
	{
		for (j = 0; j < N; j++) 
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
	int i;
	for (i = 0; i < N; i++) {
		u[i] = sin((2 * M_PI*i) / N);
	}
	matrixColumnMul(matrix, u, N, column);
}

void b_columnFiller(double* column, int N)
{
	for (int i = 0; i < N; i++)
	{
		column[i] = N + 1;
	}
}


void calcShiftsAndCount(int size, int* vecOfShifts, int* vecOfCounts, int N)
{
	if (N % size == 0)
	{
		int i;
		for (i = 0; i < size; i++) 
		{
			vecOfShifts[i] = i * (N / size) * N;
			vecOfCounts[i] = (N / size) * N;
		}
	}
	else 
	{
		int r = N % size;
		int i;
		for (i = 0; i < size; i++) 
		{
			vecOfShifts[i] = i * (N / size) * N;
			vecOfCounts[i] = (N / size) * N;
		}
		int k = 1;
		while (r > 0) 
		{
			for (i = k; i < size; i++) 
			{
				vecOfShifts[i] += N;
			}
			vecOfCounts[k] += N;
			k++;
			r--;
		}
	}
}

int main(int argc, char* argv[]) 
{
	int imode = 0;
	int size, rank;
	double t;
	bool flag = false;
	int count = 0;
	struct timeval start, end;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Status status;

	int* counts = (int*)calloc(size, sizeof(int));
	int* shifts = (int*)calloc(size, sizeof(int));
	calcShiftsAndCount(size, shifts, counts, N);
	double *matrix = (double*)calloc((N*N), sizeof(double));
	double *xn = (double*)calloc(N, sizeof(double));
	double *xn1 = (double*)calloc(N, sizeof(double));
	double *b_column = (double*)calloc(N, sizeof(double));
	double *y = (double*)calloc(N, sizeof(double));
	double *ty = (double*)calloc(N, sizeof(double));

	matrixFiller(matrix, N);

	if (imode) 
	{
		b_columnCreator(b_column, matrix, N);
	}
	else 
	{
		b_columnFiller(b_column, N);
	}

	double* partOfMatrixA = (double*)calloc(counts[rank], sizeof(double));

	MPI_Scatterv(matrix, counts, shifts, MPI_DOUBLE, partOfMatrixA, counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
	for (int i = 0; i < size; i++)
	{
		counts[i] = counts[i] / N;
		shifts[i] = shifts[i] / N;
	}

	double* vecYN_mult_partA = (double*)calloc(counts[rank], sizeof(double));
	double* tmp = (double*)calloc(counts[rank], sizeof(double));
	
	if (rank == 0) gettimeofday(&start, NULL);
	while (!flag) 
	{
		for (int i = 0; i < counts[rank]; i++)
		{
			double sum = 0;
			for (int j = 0; j < N; j++) 
			{
				sum += partOfMatrixA[i * N + j] * xn[j];
			}
			tmp[i] = sum;
		}
		MPI_Allgatherv(tmp, counts[rank], MPI_DOUBLE, y, counts, shifts, MPI_DOUBLE, MPI_COMM_WORLD);
		if (rank == 0)
		{
			count++;
			columnSub(y, b_column, N, y);
		}
		double first_part = 0, second_part = 0;
		double *temp = (double*)calloc(N, sizeof(double));
		for (int i = 0; i < counts[rank]; i++) 
		{
			double sum = 0;
			for (int j = 0; j < N; j++)
			{
				sum += partOfMatrixA[i * N + j] * xn[j];
			}
			tmp[i] = sum;
		}

		MPI_Allgatherv(tmp, counts[rank], MPI_DOUBLE, temp, counts, shifts, MPI_DOUBLE, MPI_COMM_WORLD);

		for (int i = 0; i < N; i++) 
		{
			first_part += y[i] * temp[i];
			second_part += temp[i] * temp[i];
		}
		debuf(temp, N);
		t = first_part / second_part;
		if (rank == 0) 
		{
			columnNumMul(y, t, N, ty);
			columnSub(xn, ty, N, xn1);
		}
		first_part = 0;
		second_part = 0;
		double *buf = (double*)calloc(N, sizeof(double))
			for (int i = 0; i < counts[rank]; i++) 
			{
				double sum = 0;
				for (int j = 0; j < N; j++) 
				{
					sum += partOfMatrixA[i * N + j] * xn[j];
				}
				tmp[i] = sum;
			}
		MPI_Allgatherv(tmp, counts[rank], MPI_DOUBLE, temp, counts, shifts, MPI_DOUBLE, MPI_COMM_WORLD);
		if (rank == 0) 
		{
			columnSub(temp, b_column, N, buf);
			int i;

			for (i = 0; i < N; i++)
			{
				first_part += buf[i] * buf[i];
				second_part += b_column[i] * b_column[i];
			}
			first_part = sqrt(first_part);
			second_part = sqrt(second_part);
			double final = first_part / second_part;
			if (count > 10) final *= 0.01;
			debuf(buf, N);
			if (final < e)
			{
				flag = true;
			}
			else
			{
				flag = false;
			}
			std::memcpy(xn, xn1, (N * sizeof(double)));
		}
	}
	double t2 = MPI_Wtime();
	double t3 = t2 - t1;
	double t4;
	MPI_Reduce(&t3, &t4, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
	if (rank == 0) 
	{
		printf("\n count %d \n", count);
		printf("time %lf \n", t4);
	}
	MPI_Finalize();
	return 0;
}
