#include <mpi.h>
#include <stdio>
#include <cmath>

#define size_Nx 300
#define size_Ny 300
#define size_Nz 300

double phi(double x, double y, double z) 
{
	return pow(x, 2) + pow(y, 2) + pow(z, 2);
}

double rho(double x, double y, double z) 
{
	return 6 - 1e+5 * phi(x, y, z);
}

int main(int argc, char* argv[]) 
{
	MPI_Init(&argc, &argv);

	int rank = 0;
	int size = 0;
	int shift = 0;
	double max = 0;

	int rest = size_Nx % size;

	int Nx;
	int Ny = size_Ny;
	int Nz = size_Nz;

	double Hx = (fabs(1) + fabs(-1)) / (double)(size_Nx - 1);
	double Hy = (fabs(1) + fabs(-1)) / (double)(size_Ny - 1);
	double Hz = (fabs(1) + fabs(-1)) / (double)(size_Nz - 1);

	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	double start_time = MPI_Wtime();

	if (rank < rest)
	{
		Nx = size_Nx / size + 1;
		shift = rank * Nx;
	}
	else
	{
		Nx = size_Nx / size;
		shift = rest * (Nx + 1) + (rank - rest) * Nx;
	}

	double* phi_m[2];
	phi_m[0] = newdouble[Nx * Ny * Nz];
	phi_m[1] = newdouble[Nx * Ny * Nz];

	for (int i = 0; i < Nx; ++i)
	{
		for (int j = 0; j < Ny; ++j)
		{
			for (int k = 0; k < Nz; ++k)

			{
				if (0 != i + shift && 0 != j && 0 != k && size_Nx - 1 != i + shift && size_Ny - 1 != j && size_Nz - 1 != k)
				{
					phi_m[0][i*Ny*Nz + j * Nz + k] = 0;
					phi_m[1][i*Ny*Nz + j * Nz + k] = 0;
				}
				else
				{
					phi_m[0][i*Ny*Nz + j * Nz + k] = phi(-1 + (i + shift)*Hx, -1 + j * Hy, -1 + k * Hz);
					phi_m[1][i*Ny*Nz + j * Nz + k] = phi(-1 + (i + shift)*Hx, -1 + j * Hy, -1 + k * Hz);
				}
			}
		}
	}

	double denom = (2 / pow(Hx, 2) + 2 / pow(Hy, 2) + 2 / pow(Hz, 2) + 1e+5);

	double* phi_x_lower_bound = newdouble[Ny * Nz];
	double* phi_x_upper_bound = newdouble[Ny * Nz];

	MPI_Requestreq_up[2];
	MPI_Requestreq_down[2];

	while (1)
	{
		if (size - 1 != rank)
		{
			MPI_Isend(&(phi_m[0][(Nx - 1)*Ny*Nz]), Ny*Nz, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, req_down + 0);
			MPI_Irecv(phi_x_upper_bound, Ny*Nz, MPI_DOUBLE, rank + 1, 1, MPI_COMM_WORLD, req_up + 1);
		}
		if (0 != rank)
		{
			MPI_Isend(&(phi_m[0][0]), Ny*Nz, MPI_DOUBLE, rank - 1, 1, MPI_COMM_WORLD, req_up + 0);
			MPI_Irecv(phi_x_lower_bound, Ny*Nz, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, req_down + 1);
		}

		max = 0;

		for (int i = 1; i < Nx - 1; ++i)
		{
			for (int j = 1; j < Ny - 1; ++j)
			{
				for (int k = 1; k < Nz - 1; ++k)
				{
					phi_m[1][i*Ny*Nz + j * Nz + k] = ((phi_m[0][(i + 1)*Ny*Nz + j * Nz + k]
          +  phi_m[0][(i - 1)*Ny*Nz + j * Nz + k]) / pow(Hx, 2)   + (phi_m[0][i*Ny*Nz + (j + 1)*Nz + k] 
          + phi_m[0][i*Ny*Nz + (j - 1)*Nz + k]) / pow(Hy, 2) + (phi_m[0][i*Ny*Nz + j * Nz + (k + 1)]
					+ phi_m[0][i*Ny*Nz + j * Nz + (k - 1)]) / pow(Hz, 2) - rho(-1 + (i + shift)*Hx, -1 + j * Hy, -1 + k * Hz)) /denom;

					if (fabs(phi_m[1][i*Ny*Nz + j * Nz + k] - phi_m[0][i*Ny*Nz + j * Nz + k]) > max)
					{
						max = fabs(phi_m[1][i*Ny*Nz + j * Nz + k] - phi_m[0][i*Ny*Nz + j * Nz + k]);
					}
				}
			}
		}

		if (size - 1 != rank)
		{
			MPI_Wait(req_down, MPI_STATUS_IGNORE);
			MPI_Wait(req_up + 1, MPI_STATUS_IGNORE);
		} 
		if (0 != rank)
		{
			MPI_Wait(req_up, MPI_STATUS_IGNORE);
			MPI_Wait(req_down + 1, MPI_STATUS_IGNORE);
		}

		for (intj = 1; j < Ny - 1; ++j)
		{
			for (int k = 1; k < Nz - 1; ++k)
			{
				if (0 != rank)
				{
					phi_m[1][j*Nz + k] = ((phi_m[0][Ny*Nz + j * Nz + k] + phi_x_lower_bound[j*Nz + k]) / pow(Hx, 2) 
					   + (phi_m[0][(j + 1)*Nz + k] + phi_m[0][(j - 1)*Nz + k]) / pow(Hy, 2) + (phi_m[0][j*Nz + (k + 1)]
					   + phi_m[0][j*Nz + (k - 1)]) / pow(Hz, 2) - rho(-1 + shift * Hx, -1 + j * Hy, -1 + k * Hz)) / denom;

					if (fabs(phi_m[1][j*Nz + k] - phi_m[0][j*Nz + k]) > max)
					{
						max = fabs(phi_m[1][j*Nz + k] - phi_m[0][j*Nz + k]);
					}
				}


				if (size - 1 != rank)
				{
					phi_m[1][(Nx - 1)*Ny*Nz + j * Nz + k] =	((phi_x_upper_bound[j*Nz + k] 
          + phi_m[0][(Nx - 2)*Ny*Nz + j * Nz + k]) / pow(Hx, 2) + (phi_m[0][(Nx - 1)*Ny*Nz + (j + 1)*Nz + k] 
          + phi_m[0][(Nx - 1)*Ny*Nz + (j - 1)*Nz + k]) / pow(Hy, 2)+ (phi_m[0][(Nx - 1)*Ny*Nz + j * Nz + (k + 1)]  
          + phi_m[0][(Nx - 1)*Ny*Nz + j * Nz + (k - 1)]) / pow(Hz, 2) - rho(-1 + (Nx - 1 + shift)*Hx, -1 + j * Hy, -1 + k * Hz)) / denom;

					if (fabs(phi_m[1][(Nx - 1)*Ny*Nz + j * Nz + k] - phi_m[0][(Nx - 1)*Ny*Nz + j * Nz + k]) > max)
					{
						max = fabs(phi_m[1][(Nx - 1)*Ny*Nz + j * Nz + k] - phi_m[0][(Nx - 1)*Ny*Nz + j * Nz + k]);
					}
				}
			}
		}

		swap(phi_m[0], phi_m[1]);

		double max_tmp;

		MPI_Allreduce(&max, &max_tmp, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

		if (1e-8 > max_tmp)
		{
			break;
		}
	}

	max = 0;

	for (int i = 0; i < Nx; ++i)
	{
		for (int j = 1; j < Ny - 1; ++j)
		{
			for (int k = 1; k < Nz - 1; ++k)
			{
				if (i + shift != size_Nx - 1 && i + shift != 0)
				{
					if (fabs(phi_m[0][i*Ny*Nz + j * Nz + k] - phi(-1 + (i + shift)*Hx, -1 + j * Hy, -1 + k * Hz)) > max)
					{
						max = fabs(phi_m[0][i*Ny*Nz + j * Nz + k] - phi(-1 + (i + shift)*Hx, -1 + j * Hy, -1 + k * Hz));
					}
				}
			}
		}
	}

	double max_delta;

	MPI_Reduce(&max, &max_delta, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

	delete[] phi_m[0];
	delete[] phi_m[1];
	delete[] phi_x_upper_bound;
	delete[] phi_x_lower_bound;

	double finish_time = MPI_Wtime();

	if (0 == rank) 
	{
		printf("Time: %lf\n", finish_time - start_time);
	}

	MPI_Finalize();
	return 0;
}
