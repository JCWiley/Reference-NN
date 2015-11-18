/*
************ RANDOM.H ***********

This include file contains routines for getting a uniformly distributed
random variable in the interval [0,1], a gaussian
distributed random variable with zero mean and unity variance,
an exponential distributed random variable, and a
and a cauchy distributed random variable with

To use them (x & y declared as floats), 

x = uniform(&idum);   
y = normal(&idum);
z = cauchy(&idum);
u = expdev(&idum);

Notice that a pointer to "idum" is used.


The file also contains routines for getting uniformly
generated integers over specified intervals.

Before you use any of these routines, you must initialize
the long variable "idum" (declared in this file).  To do this
you must include in your main program the following:

#include <time.h>

int main(void )
{
.
.
.

srand((unsigned) time(NULL));
idum = -rand();

}


*/

#include <math.h>

#define IA	16807
#define IM	2147483647
#define AM	(1.0/IM)
#define IQ	127773
#define IR	2836
#define NTAB	32
#define NDIV	(1+(IM-1)/NTAB)
#define EPSILON	1.2e-7
#define RNMX	(1.0-EPSILON)

//random between 0 and 1
float uniform(long *idum);

float normal(long *idum);

float expdev(long *idum);
/* gives cachy r.v. centered about 0 with pdf 
f(x) = 1/(pi*(1+x*x)). */
double cauchy(long *idum);
/*  returns a random integer in the interval [0,n-1] */
int r0n(int n);
/*  returns a random integer in the interval [n,m] */
int rnm(int n, int m);