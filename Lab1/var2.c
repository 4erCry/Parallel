#include <iostream> 
#include <cmath> 
#include <sys/time.h> 
#include <cstdlib> 
#include <cstring> 
#include <stdio.h> 
#include <mpi.h>

#define e 1e-8  

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

void mpiMatrixFiller(double *matrix_part, int rank, int part_size, int N, int start_column)
{
	for (int i = 0; i < part_size; i++) 
	{
		for (int j = 0; j < N; j++)
		{
			if (start_column != j)
			{
				matrix_part[j * N + i] = 1.0;
			}
			else 
			{
				matrix_part[j * N + i] = 2.0;
			}
		}
		start_column++;
	}
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
	int N = 15000;
	int size, rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Status status;
	int i, j;
	int part_size;
	int start_string;
	int* counts = (int*)malloc(size * sizeof(int));
	int* shifts = (int*)malloc(size * sizeof(int));
	calcShiftsAndCount(size, shifts, counts, N);
	double *b_column = (double*)calloc(N, sizeof(double));
	double *xn = (double*)calloc(N, sizeof(double));

	if (rank == 0) 
	{
		b_columnFiller(b_column, N);
	}
	
	if (rank >= N % size) 
	{
		part_size = N / size;
		start_string = (((N % size) * (N / size + 1)) + (rank - N % size) * (N / size));
	}
	else 
	{
		part_size = N / size + 1;
		start_string = rank * (N / size + 1);
	}

	double *matrix_part = (double*)calloc((N*part_size), sizeof(double));

	mpiMatrixFiller(matrix_part, rank, part_size, N, start_string);

	double* partOfVectorB = (double*)malloc((counts[rank] / N) * sizeof(double));
	for (i = 0; i < size; i++) 
	{
		counts[i] = counts[i] / N;
		shifts[i] = shifts[i] / N;
	}
	MPI_Scatterv(b_column, counts, shifts, MPI_DOUBLE, partOfVectorB, counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

	double *xn1 = (double*)calloc(N, sizeof(double));
	double *buffer = (double*)calloc(N, sizeof(double));
	double *y = (double*)calloc(N, sizeof(double));
	double *ty = (double*)calloc(N, sizeof(double));

	double *partOfbuffer = (double*)calloc(counts[rank], sizeof(double));
	double *partofy = (double*)calloc(counts[rank], sizeof(double));
	double *partOfXn = (double*)calloc(counts[rank], sizeof(double));
	double *partOfXn1 = (double*)calloc(counts[rank], sizeof(double));
	MPI_Scatterv(xn, counts, shifts, MPI_DOUBLE, partOfXn, counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatterv(xn1, counts, shifts, MPI_DOUBLE, partOfXn1, counts[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

	double t;
	bool flag = false;
	int count = 0;
	double t1 = MPI_Wtime();
	double normB = 0;
	double localnormB = 0;
	for (i = 0; i < counts[rank]; i++) 
	{
		localnormB += partOfVectorB[i] * partOfVectorB[i];
	}
	MPI_Allreduce(&localnormB, &normB, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

	normB = sqrt(normB);
	while (!flag) 
	{
		count++;
		debuf(buffer, N);
		for (i = 0; i < counts[rank]; i++) 
		{
			for (j = 0; j < N; j++)
			{
				partofy[i] += partOfMatrixA[j*N + i] * partOfXn[i];
			}

		}
		for (i = 0; i < counts[rank]; i++)
		{
			partofy[i] -= -partOfVectorB[i];
		}
		MPI_Allgatherv(partofy, counts[rank], MPI_DOUBLE, y, counts, shifts, MPI_DOUBLE, MPI_COMM_WORLD);

		double first_part = 0, second_part = 0;
		double *temp = (double*)calloc(N, sizeof(double));
		double *partOftemp = (double*)calloc(counts[rank], sizeof(double));
		for (i = 0; i < counts[rank]; i++) 
		{
			for (j = 0; j < N; j++)
			{
				partOftemp[i] += partOfMatrixA[j*N + i] * y[i];
			}
		}

		double local_first_part = 0, local_second_part = 0;
		for (i = 0; i < counts[rank]; i++) 
		{
			local_first_part += partofy[i] * partOftemp[i];
			local_second_part += partOftemp[i] * partOftemp[i];
		}
		MPI_Allreduce(&local_first_part, &first_part, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		MPI_Allreduce(&local_second_part, &second_part, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		debuf(temp, N);
		t = first_part / second_part;
		for (i = 0; i < counts[rank]; i++)
		{
			partOfXn1[i] = partOfXn[i] - partofy[i] * t;
		}
		MPI_Allgatherv(partOfXn, counts[rank], MPI_DOUBLE, xn, counts, shifts, MPI_DOUBLE, MPI_COMM_WORLD);
		MPI_Allgatherv(partOfXn1, counts[rank], MPI_DOUBLE, xn1, counts, shifts, MPI_DOUBLE, MPI_COMM_WORLD);
		double totaldev = 0;
		double localdevidendForCheck = 0;
		for (i = 0; i < counts[rank]; i++) 
		{
			double sum = 0;
			for (j = 0; j < N; j++) 
			{
				sum += partOfMatrixA[j * N + i] * xn1[i];
			}
			localdevidendForCheck += (sum - partOfVectorB[i]) * (sum - partOfVectorB[i]);
		}
		MPI_Allreduce(&localdevidendForCheck, &totaldev, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
		double final = sqrt(totaldev) / normB;
		if (count > 10) final *= 0.01;
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
	double t2 = MPI_Wtime();
	double t3 = t2 - t1;
	double t4;
	MPI_Reduce(&t3, &t4, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

	if (rank == 0) 
	{
		printf("\n count %d \n", count);
		printf("\n time %lf \n", t4);
	}
	MPI_Finalize();
	return 0;
}
