#ifndef DNAKERNELMETA_CUH
#define DNAKERNELMETA_CUH

#include "DNAList.h"
#include "global.h"


extern __constant__  int neighborindex[27];
extern __constant__ float min1, min2, min3, max1, max2, max3;

extern  __constant__  float d_rDNA[72];

__device__ float caldistanceMeta(float3 pos1, float3 pos2);
__device__ float3 pos2localMeta(int type, float3 pos, int index);
__global__ void phySearchMeta(
	int num, 
	Edeposit* d_edrop, 
	int* dev_chromatinIndex,
	int* dev_chromatinStart,
	int* dev_chromatinType, 
	CoorBasePair* dev_straightChrom,
	CoorBasePair* dev_segmentChrom,
	CoorBasePair* dev_bendChrom,
	float3* dev_straightHistone,
	float3* dev_bendHistone, 
	combinePhysics* d_recorde,
	float3 *dev_chromosome, 
	int *dev_chromosome_type,
	int *dev_segmentIndex,
	int *dev_segmentStart, 
	int *dev_segmentType);
  
__global__ void chemSearchMeta(int num,
  Edeposit* d_edrop, 
	int* dev_chromatinIndex,
	int* dev_chromatinStart,
	int* dev_chromatinType, 
	CoorBasePair* dev_straightChrom, 
	CoorBasePair* dev_segmentChrom,
	CoorBasePair* dev_bendChrom,
	float3* dev_straightHistone,
	float3* dev_bendHistone, 
	combinePhysics* d_recorde,
	float3 *dev_chromosome, 
	int *dev_chromosome_type,
	int *dev_segmentIndex, 
	int *dev_segmentStart, 
	int *dev_segmentType);
#endif
