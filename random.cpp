#include "random.h"

long idum;

float uniform(long *idum)
{
	int j;
	long k;
	static long iy=0;
	static long iv[NTAB];
	float temp;

	if(*idum <= 0 || !iy)
	{
		if(-(*idum) < 1) *idum=1;
		else
			*idum= -(*idum);
		for(j=NTAB+7; j>=0; j--)
		{
			k=(*idum)/IQ;
			*idum=IA*(*idum-k*IQ)-IR*k;
			if(*idum<0) *idum += IM;
			if(j<NTAB) iv[j]= *idum;
		}
		iy=iv[0];
	}

	k=(*idum)/IQ;
	*idum=IA*(*idum-k*IQ)-IR*k;
	if(*idum<0) *idum += IM;
	j=(int )iy/NDIV;
	iy=iv[j];
	iv[j]= *idum;
	if((temp=AM*iy)>RNMX) return RNMX;
	else return temp;
}

float normal(long *idum)
{
	static int iset=0;
	static float gset;
	float fac,rsq,v1,v2;

	if(iset==0)
	{
		do
		{
			v1=2.0*uniform(idum)-1.0;
			v2=2.0*uniform(idum)-1.0;
			rsq=v1*v1+v2*v2;
		}while(rsq>=1.0 || rsq==0.0);
		fac=sqrt(-2.0*log(rsq)/rsq);
		gset=v1*fac;
		iset=1;
		return v2*fac;
	}
	else
	{
		iset=0;
		return gset;
	}
}

float expdev(long *idum)
{
	float dum;

	do
	dum= uniform(idum );
	while (dum==0.0);
	return -log(dum);
}

double cauchy(long *idum)
{
	double z1,z2;

	z1= normal(idum );
	z2= normal(idum );
	if(z2 != 0.0)
		return (z1/z2);
	else
		return 0.0;
}

int r0n(int n)
{
	return (int )floor((float )n*uniform(&idum));
}

int rnm(int n, int m)
{
	int k;

	do
	{
		k=r0n(m+1);
	}while(k<n || k>m);
	return k;
}

