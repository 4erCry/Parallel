#define e 1e-10

void matrixFiller(double *matrix, int N)
{
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

	for (int i = 0; i < N; i++) 
	{
		u[i] = sin((2 * M_PI*i) / N);
	} 

	for (int i = 0; i < N; i++)
	{
		for (int j = 0; j < N; j++)
		{
			column[i] += matrix[i*N + j] * u[j];
		}
	}
}

void b_columnFiller(double* column, int N)
{
	for (int i = 0; i < N; i++)
	{
		column[i] = N + 1;
	}
}

int main()
{
	int N = 17000;
	int i, j;
	double t;
	bool flag = false;
	int count = 0;
          int imode = 1;
	struct timeval start, end;
	double *matrix = (double*)calloc((N*N), sizeof(double));
	double *xn = (double*)calloc(N, sizeof(double));
	double *xn1 = (double*)calloc(N, sizeof(double));
	double *buffer = (double*)calloc(N, sizeof(double));
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
	
	gettimeofday(&start, NULL);

#pragma omp parallel shared(flag)
	{
		while (!flag) 
		{
			count++;

#pragma omp for
			for (i = 0; i < N; i++)
			{
				buffer[i] = 0;
			}
			
#pragma omp for
			for (i = 0; i < N; i++) 
			{
				for (j = 0; j < N; j++) 
				{
					buffer[i] += matrix[i*N + j] * xn[j];
				}
			}

#pragma omp for
			for (i = 0; i < N; i++)
			{
				y[i] = buffer[i] - b_column[i];
			}
 
			double first_part = 0, second_part = 0;
			double *temp = (double*)calloc(N, sizeof(double));

#pragma omp for
			for (i = 0; i < N; i++) 
			{
				for (j = 0; j < N; j++)
				{
					temp[i] += matrix[i*N + j] * y[j];
				}
			}

#pragma omp for
			for (i = 0; i < N; i++)
			{
				first_part += y[i] * temp[i];
			}

#pragma omp for
			for (i = 0; i < N; i++) 
			{
				second_part += temp[i] * temp[i];
			}
#pragma omp for
			for (i = 0; i < N; i++) 
			{
				temp[i] = 0;
			}

			t = first_part / second_part;

#pragma omp for
			for (i = 0; i < N; i++)
			{
				ty[i] = y[i] * t;
			}

#pragma omp for
			for (int i = 0; i < N; i++)
			{
				xn1[i] = xn[i] - ty[i];
			}

			double *buf = (double*)calloc(N, sizeof(double));
			double *buf2 = (double*)calloc(N, sizeof(double));

#pragma omp for
			for (i = 0; i < N; i++) 
			{
				for (j = 0; j < N; j++)
				{
					buf[i] += matrix[i*N + j] * xn[j];
				}
			}

#pragma omp for
			for (i = 0; i < N; i++)
			{
				buf2[i] = buf[i] - b_column[i];
			}

#pragma omp for		
			for (int i = 0; i < N; i++)
			{
				first_part += buf2[i] * buf2[i];
			}

#pragma omp for		
			for (int i = 0; i < N; i++) 
			{
				second_part += b_column[i] * b_column[i];
			}

			first_part = sqrt(first_part);
			second_part = sqrt(second_part);
			double final = first_part / second_part;
			if (count > 10) final *= 0.01;

#pragma omp for
			for (i = 0; i < N; i++) 
			{
				buf[i] = 0;
			}

#pragma omp for
			for (i = 0; i < N; i++) 
			{
				buf2[i] = 0;
			}

			if (final < e) 
			{
				flag = true;
			}
			else 
			{
				flag = false;
			}
			memcpy(xn, xn1, (N * sizeof(double)));
		}
	}
	gettimeofday(&end, NULL);
	double dt_sec = (end.tv_sec - start.tv_sec);
	double dt_usec = (end.tv_usec - start.tv_usec);
	double dt = dt_sec + 0.000001 * dt_usec;
	printf("time %lf \n", dt);
	return 0;
}
