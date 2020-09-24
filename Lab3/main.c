#include <stdio.h>
#include <mpi.h>

#define A_height 1500
#define A_width 1500 
#define B_width 1500 

void create_matrix(double* matrixA, double* matrixB)
{
	for (int i = 0; i < A_height; ++i)
	{
		for (int j = 0; j < A_width; ++j) 
		{
			matrixA[i * A_width + j] = 1;
		}
	}
	
	for (int i = 0; i < A_width; ++i)	
	{
		for (int j = 0; j < B_width; ++j) 
		{
			matrixB[i * B_width + j] = 1;
		}
	}
}

void typesCreate(MPI_Datatype* Bb, MPI_Datatype* Cc, int rowsPerProc, int colsPerProc)
{
	MPI_Type_vector(A_width, colsPerProc, B_width, MPI_DOUBLE, Bb);
	MPI_Type_vector(rowsPerProc, colsPerProc, B_width, MPI_DOUBLE, Cc);

	MPI_Type_create_resized(*Bb, 0, colsPerProc * sizeof(double), Bb);
	MPI_Type_create_resized(*Cc, 0, colsPerProc * sizeof(double), Cc);

	MPI_Type_commit(Bb);
	MPI_Type_commit(Cc);
}

void createComms(MPI_Comm commDek, MPI_Comm* columns, MPI_Comm* rows) 
{
	int onlyRows[2] = { 1,0 };
	int onlyCols[2] = { 0,1 };

	MPI_Cart_sub(commDek, onlyCols, columns);
	MPI_Cart_sub(commDek, onlyRows, rows);
}


void main(int argc, char* argv[]) 
{
	MPI_Init(&argc, &argv);

	int count, rank;

	double* A;
	double* B;
	double* C;

	MPI_Comm_size(MPI_COMM_WORLD, &count);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int dims[2] = { 0, 0 };
	int periods[2] = { 0, 0 };
	MPI_Comm commDek;
	MPI_Dims_create(count, 2, dims);
	MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &commDek);
	
	if (rank == 0)
	{
		A = (double*)calloc(A_height * A_width, sizeof(double));
		B = (double*)calloc(B_width * A_width, sizeof(double));
		C = (double*)calloc(B_width * A_height, sizeof(double));

		create_matrix(A, B);
	}

	double start_time = MPI_Wtime();

	MPI_Datatype Bb;
	MPI_Datatype Cc;

	int* sendCountsB = NULL;
            int* sendCountsC = NULL;
	int* displsB = NULL;
	int* displsC = NULL;

	int coords[2];
	int commSize;

	int rowsPerProc = A_height / dims[0];
	int colsPerProc = B_width / dims[1];

	MPI_Comm_size(commDek, &commSize);
	MPI_Cart_coords(commDek, rank, 2, coords);

	double* aPart = (double*)calloc(rowsPerProc * A_width, sizeof(double));
	double* bPart = (double*)calloc(colsPerProc * A_width, sizeof(double));

	double* cPart = (double*)calloc(colsPerProc * rowsPerProc, sizeof(double));

	if (rank == 0)
	{
		typesCreate(&Bb, &Cc, rowsPerProc, colsPerProc);

		*sendCountsB = (int*)calloc(dims[1], sizeof(int));
		*sendCountsC = (int*)calloc(commSize, sizeof(int));
		*displsB = (int*)calloc(dims[1], sizeof(int));
		*displsC = (int*)calloc(commSize, sizeof(int));

		for (int i = 0; i < dims[1]; ++i)
		{
			(*displsB)[i] = i;
			(*sendCountsB)[i] = 1;
		}

		for (int i = 0; i < dims[0]; ++i)
		{
			for (int j = 0; j < dims[1]; ++j)
			{
				(*displsC)[i * dims[1] + j] = i * dims[1] * rowsPerProc + j;
			}
		}

		for (int i = 0; i < commSize; ++i)
		{
			(*sendCountsC)[i] = 1;
		}
	}

	MPI_Comm commCol;
	MPI_Comm commRow;

	createComms(commDek, &commCol, &commRow);

	if (coords[1] == 0)
	{
		MPI_Scatter(A, rowsPerProc * A_width, MPI_DOUBLE, aPart, rowsPerProc * A_width, MPI_DOUBLE, 0, commCol);
	}

	if (coords[0] == 0)
	{
		MPI_Scatterv(B, sendCountsB, displsB, Bb, bPart, colsPerProc * A_width, MPI_DOUBLE, 0, commRow);
	}

	MPI_Bcast(aPart, rowsPerProc * A_width, MPI_DOUBLE, 0, commRow);
	MPI_Bcast(bPart, colsPerProc * A_width, MPI_DOUBLE, 0, commCol);

	for (int i = 0; i < rowsPerProc; ++i) 
	{
		for (int j = 0; j < colsPerProc; ++j)
		{
			for (int k = 0; k < A_width; ++k)
			{
				cPart[i * colsPerProc + j] += aPart[i * A_width + k] * bPart[k * colsPerProc + j];
			}
		}
	}


	MPI_Gatherv(cPart, colsPerProc * rowsPerProc, MPI_DOUBLE, C, sendCountsC, displsC, Cc, 0, commDek);

	if (rank == 0)
	{
		free(sendCountsB);
		free(displsB);
		free(sendCountsC);
		free(displsC);

		MPI_Type_free(&Bb);
		MPI_Type_free(&Cc);
	}

	free(aPart);
	free(bPart);
	free(cPart);

	if (rank == 0) 
	{
		free(A);
		free(B);
		free(C);
	}

	double finish_time = MPI_Wtime();

	printf("Time: %lf\n", finish_time - start_time);

	MPI_Finalize();
}
