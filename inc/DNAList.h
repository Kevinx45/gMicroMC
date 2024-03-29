#ifndef DNALIST_H
#define DNALIST_H

#include "global.h"
#include <algorithm>

#define NUCLEUS_DIM 200 //# of bins
#define STRAIGHT_BP_NUM 5040
#define BEND_BP_NUM 4041
#define BEND_HISTONE_NUM 24
#define STRAIGHT_HISTONE_NUM 30
#define UNITLENGTH 55 // size of a voxel in nm

//metaphase DNA structure variables
#define NUCLEUS_DIM_META 67 //# of bins cylinder
#define NUCLEUS_DIM_Z_META 42 // height of the cylinder
#define STRAIGHT_BP_NUM_META 200
#define BEND_BP_NUM_META 200
#define SEGMENT_BP_NUM_META 17
#define BEND_HISTONE_NUM_META 1
#define STRAIGHT_HISTONE_NUM_META 1
#define UNITLENGTH_META 11
#define NUMCHROMOSOMES_META 46
#define CYLINDERRADIUS_META ((NUCLEUS_DIM_META * UNITLENGTH_META) / 2)
#define CYLINDERHEIGHT_META (NUCLEUS_DIM_Z_META * UNITLENGTH_META)
#define TOTALBP_META 25633020
#define MAXNUMPAR_META 131072 //1048576 //524288 // maximum particles at one time
#define MAXNUMPAR2_META MAXNUMPAR_META*3 //maximum particles to be stored on GPU (current particles including dead ones and new ones in a reaction)
#define MAXNUMNZBIN_META 2000000 //maximum number of non-zero bins, overlap 


#define EMIN 18
#define EMAX 18
#define PROBCHEM 0.1
#define DiffusionOfOH 2.8 //  10^9 nm*nm/s
#define SPACETOBODER 2
#define RBASE 0.5
#define RHISTONE 3.13
#define RSUGAR 0.431
#define RPHYS 0.1
#define dDSB 10
#define dS 216

typedef struct
{ 
    float3 base, right, left;
} CoorBasePair;

typedef struct
{ 
    int index;
    int boxindex;
    float3 position;
    float3 dir;
} DNAsegment;

typedef struct
{ 
    float e;
    float3 position;
} Edeposit;

typedef struct
{
	int x, y, z, w;//DNA index, base index, left or right, damage type
}chemReact;

typedef struct
{
	chemReact site;
	float prob1,prob2;
}combinePhysics;

class DNAList
{
public:
    DNAList();
    virtual ~DNAList();
    void initDNA();
    void initDNAMeta();
    void saveResults();
    void calDNAreact_radius(float* diffCoeff);
    Edeposit* readStage(int *numPhy,int mode, const char* fname);
    void quicksort(chemReact*  hits,int start, int stop, int sorttype);
    chemReact* combinePhy(int* totalphy, combinePhysics* recorde,int mode);
    void damageAnalysis(int counts, chemReact* recordpos,float totaldose,int idle1,int idle2);
    void run();
public:
    float rDNA[12]={0};
    float totaldose = 0;
    int complexity[14]={0,0,0,0,0,0,0,0,0,0,0,0,0,0};//SSB,2xSSB, SSB+, 2SSB, DSB, DSB+, DSB++
	int results[7]={0,0,0,0,0,0,0};//SSBd, SSbi, SSbm, DSBd, DSBi, DSBm, DSBh.
    //GPU
    CoorBasePair *dev_bendChrom, *dev_straightChrom, *dev_segmentChrom;
    float3 *dev_bendHistone, *dev_straightHistone, *dev_chromosome;
    int *dev_chromatinIndex, *dev_chromatinStart, *dev_chromatinType, *dev_chromosome_type, *dev_segmentIndex, *dev_segmentStart, *dev_segmentType;
};

struct compare_dnaindex
{
    __host__ __device__ bool operator()(chemReact a, chemReact b)
    {
        return a.x < b.x;
    }
};
struct compare_baseindex
{
    __host__ __device__ bool operator()(chemReact a, chemReact b)
    {
        return a.y < b.y;
    }
};
struct compare_boxindex
{
    __host__ __device__ bool operator()(combinePhysics a, combinePhysics b)
    {
        return a.site.x < b.site.x;
    }
};

#endif
