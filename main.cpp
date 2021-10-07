#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include "mpi.h"

using namespace std;


const int numberOfIterations = 1e6;
const int printFrequincy = 10; // Iterations per output.
const double timeStep = 0.001; // Seconds.

const double B = 0.25;
const double U = -5;
const double R = 0.002;
const double Z = 0.002;
const double k = 8987551787.3681764;
const double m = 75.921402618*1.6605e-27;
const double q = 27*1.60217662e-19;
const double Wm = -2*U/(R*R+2*Z*Z);
const double Wc = q*B/m - Wm;
const double Wax = sqrt(2*q*Wm/m);

class Ion {
public:
	int str;
	double coordinates[3]; // Current coordinates [x, y, z].
	double velocity[3]; // Velocity values [Vx, Vy, Vz].
	double acceleration[3]; // Aacceleration of ion [ax, ay, az].

	Ion(double Vx, double Vy, double Vz,
			double x=0, double y=0, double z=0)
	{
		this->coordinates[0] = x;
		this->coordinates[1] = y;
		this->coordinates[2] = z;
		this->velocity[0] = Vx;
		this->velocity[1] = Vy;
		this->velocity[2] = Vz;
	}

	void refreshAcceleration()
	{
		for (int i = 0; i < 3; ++i) acceleration[i] = 0;
	}
};

void CoulombForce (Ion &ionA, double* ionB) { // Calculates Coulomb force for two ions.
	double residuals[3]; // Residuals of corresponding coordinates. 
	double ssr; // Sum of squared residuals.
	for (int i = 0; i < 3; ++i) {
		residuals[i] = (ionB[i] - ionA.coordinates[i]);
		ssr += residuals[i]*residuals[i];
	}

	double r = sqrt(ssr);
	double F = k*q*q/(r*r);

	for (int i = 0; i < 3; ++i) {
		double forceProjection = (residuals[i] / r) * F;
		ionA.acceleration[i] = forceProjection / m;
	}
}

void forOtherCoulomb(Ion & ion, double** other, int size, int id) //Calculates Coulomb force caused by other ions.
{
	ion.refreshAcceleration();

	MPI_Status stat;

	for (int i = 0; i < 3; ++i) {
		other[id][i] = ion.coordinates[i];
	}

// Begining of thread communication block.
// This is need to get current location of other ions.
	
	for (int i = 0; i < size; ++i) {
		if (i != id) {
			MPI_Send(other[id], 3, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
		}
	}

	for (int i = 0; i < size; ++i) {
		if(i != id) {
			MPI_Recv(other[i], 3, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &stat);
		}
	}
// End of thread communication block.


	for (int i = 0; i < size; ++i) {
		if (i != id) {
			CoulombForce(ion, other[i]);
		}
	}

	return;
}

// Calculates acceleration.
// Formulas are the motion laws for penning trap plus acceleration caused by Coulumb force.
void PenningMotion(Ion &ion, /*double *a,*/ double *coordinates, double *velocity) {
	/*a[0] = (q*B*(ion.velocity[1] + velocity[1]) + (ion.coordinates[0]+coordinates[0])*q*Wm)/m + ion.acceleration[0];
	a[1] = (-q*B*(ion.velocity[0] + velocity[0]) + (ion.coordinates[1]+coordinates[1])*q*Wm)/m + ion.acceleration[1];
	a[2] = (-2*(ion.coordinates[2]+coordinates[0])*q*Wm)/m + ion.acceleration[2];*/
	ion.acceleration[0] += (q*B*(ion.velocity[1] + velocity[1]) + (ion.coordinates[0] + coordinates[0])*q*Wm) / m;
	ion.acceleration[1] += (-q*B*(ion.velocity[0] + velocity[0]) + (ion.coordinates[1] + coordinates[1])*q*Wm) / m;
	ion.acceleration[2] += (-2*(ion.coordinates[2] + coordinates[0])*q*Wm) / m;
}

void RungeKutta(Ion &ion)
{
	double dT = timeStep / Wc;
	const double coefs[4] = {1, 0.5, 0.5, 1};
	// double a[3] = {0};
	double k[4][3] = {0}, l[4][3] = {0};
	
	for(int i = 0; i < 4; ++i) {
		double lPrev[3], kPrev[3];
		for (int j = 0; j < 3; ++j) {
			lPrev[j] = l[i-1][j] * coefs[i];
			kPrev[j] = k[i-1][j] * coefs[i];
		}
		
		PenningMotion(ion, /*a,*/ lPrev, kPrev);
		for(int j = 0; j < 3; ++j) {
			k[i][j] = dT * ion.acceleration[j];
			l[i][j] = dT * ion.velocity[j];
		}
	}

	for (int i = 0; i < 3; ++i) {
		ion.velocity[i] += (k[0][i] + 2*k[1][i] + 2*k[2][i] + k[3][i]) / 6;
		ion.coordinates[i] += (l[0][i] + 2*l[1][i] + 2*l[2][i] + l[3][i]) / 6;
	}
}

int main(int argc, char** argv)
{
	int size, id;

	MPI_Init(&argc, &argv); // Initialize parallel part.
	
	MPI_Comm_size(MPI_COMM_WORLD, &size); // Get information about current thread.
	MPI_Comm_rank(MPI_COMM_WORLD, &id);

	srand(time(nullptr)*id);

	string fname = "output/output" + to_string(id) + ".txt";
	ofstream out (fname, ios_base::out);

	Ion ion (0, 0, 0);

	double** otherIons = new double*[size];
	for (int i = 0; i < size; ++i) {
		otherIons[i] = new double[3];
	}
	
	for (int i = 0; i < 3; ++i) {
		ion.coordinates[i] = rand()%100*(10e-6);
	}
	out << "Time, s;x, m;y, m;z, m\n";
	for(int i = 0; i <= numberOfIterations; ++i) {
		if(i % printFrequincy == 0) {
			out << timeStep / Wc * i << ';';
			out << ion.coordinates[0] << ';' << ion.coordinates[1] << ';' << ion.coordinates[2] << '\n';
		}
		forOtherCoulomb(ion, otherIons, size, id);
		RungeKutta(ion);
	}

	MPI_Finalize();
	cout << "Programm finished\n";

	return 0;
}
