#ifndef DNAKERNELMETA_CUH
#define DNAKERNELMETA_CUH

#include "microMC_chem.h"
#include "global.h"


extern __constant__  int neighborindex[27];
extern __constant__ float min1, min2, min3, max1, max2, max3;

extern  __constant__  float d_rDNA[72];

#endif
