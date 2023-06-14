#ifndef __REALTIME__
#define __REALTIME__
#include <stdio.h>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <string>
#include "microMC_chem.h"

#define REALTIME_FILEIN "./Input/electron_broadspectrum_2022_09_19/"
#define REALTIME_FILEOUT "/home/satzhan/repos/gMicroMC/chem_stage/Results/electron_broadspectrum_2022_09_19_v20/totalrecordMETAtot_a"

// #define REALTIME_FILEIN "./Input/data80proton/"
// #define REALTIME_FILEOUT "/home/satzhan/repos/gMicroMC/chem_stage/Results/data80proton_v13_us_a/totalrecordMETAtot"
#define FILEOH "/OH_1ns"
#define FILEOHNAME "1ns"

void printDevProp(int device)
//      print out device properties
{
    int devCount;
    cudaDeviceProp devProp;
//      device properties

    cudaGetDeviceCount(&devCount);
	cout << "Number of device:              " << devCount << endl;
	cout << "Using device #:                " << device << endl;
    cudaGetDeviceProperties(&devProp, device);
	
	printf("Major revision number:         %d\n",  devProp.major);
    printf("Minor revision number:         %d\n",  devProp.minor);
    printf("Name:                          %s\n",  devProp.name);
    printf("Total global memory:           %7.2f MB\n",  
	devProp.totalGlobalMem/1024.0/1024.0);
    printf("Total shared memory per block: %5.2f kB\n",  
	devProp.sharedMemPerBlock/1024.0);
    printf("Total registers per block:     %u\n",  devProp.regsPerBlock);
    printf("Warp size:                     %d\n",  devProp.warpSize);
    printf("Maximum memory pitch:          %lu\n",  devProp.memPitch);
    printf("Maximum threads per block:     %d\n",  devProp.maxThreadsPerBlock);
    	
	printf("Maximum dimension of block:    %d*%d*%d\n", 			
	devProp.maxThreadsDim[0],devProp.maxThreadsDim[1],devProp.maxThreadsDim[2]);
	printf("Maximum dimension of grid:     %d*%d*%d\n", 
	devProp.maxGridSize[0],devProp.maxGridSize[1],devProp.maxGridSize[2]);
    printf("Clock rate:                    %4.2f GHz\n",  devProp.clockRate/1000000.0);
    printf("Total constant memory:         %5.2f kB\n",  devProp.totalConstMem/1024.0);
    printf("Texture alignment:             %lu\n",  devProp.textureAlignment);
    printf("Concurrent copy and execution: %s\n",  (devProp.deviceOverlap ? "Yes" : "No"));
    printf("Number of multiprocessors:     %d\n",  devProp.multiProcessorCount);
    printf("Kernel execution timeout:      %s\n",  (devProp.kernelExecTimeoutEnabled ? "Yes" : "No"));
//      obtain computing resource

}

void calDNAreact_radius(float* rDNA,float deltat)
{
	float k[5]={6.1,9.2,6.4,6.1,1.8};
	float tmp=sqrtf(PI*DiffusionOfOH*deltat*0.001);
	for(int i=0;i<5;i++)
	{
		rDNA[i]=k[i]/(4*PI*DiffusionOfOH)*10/6.023;//k 10^9 L/(mol*s), Diffusion 10^9 nm^2/s. t ps
		rDNA[i]=sqrtf(rDNA[i]*tmp+tmp*tmp*0.25)-tmp*0.5;
	}
	rDNA[5]=0;//histone protein absorption radius, assumed!!!
}

__device__ float caldistance(float3 pos1, float3 pos2)
{
	return (sqrtf((pos1.x -pos2.x)*(pos1.x -pos2.x)+(pos1.y -pos2.y)*(pos1.y -pos2.y)+(pos1.z -pos2.z)*(pos1.z -pos2.z)));
}
// accessors and mutators functions 
// can be names like variables
// is this accessor? or mutator?
// this function looks like it does something more? 
__device__ float3 PosToWall(int type, float3 pos, int index) 
{
	// xy xz yz
	float shiftz;
	float shifty;
	float shiftx;
	
	if (0 <= index && index < 4) { // xy
		// shift by -z
		shiftz = -5.5;
		// array <int, 2> subs = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
		if (index % 4 == 0) {
			shiftx = 0.0;
			shifty = 5.5/2.0;
		}	
		if (index % 4 == 1) {
			shiftx = 5.5/2.0;
			shifty = 0;
		}	
		if (index % 4 == 2) {
			shiftx = 0.0;
			shifty = -5.5/2.0;
		}	
		if (index % 4 == 3) {
			shiftx = -5.5/2.0;
			shifty = 0;
		}	
	}
	if (4 <= index && index < 8) { // xz
		// shift by -y
		shifty = -5.5;
		// array <int, 2> subs = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
		if (index % 4 == 0) {
			shiftx = 0.0;
			shiftz = 5.5/2.0;
		}	
		if (index % 4 == 1) {
			shiftx = 5.5/2.0;
			shiftz = 0;
		}	
		if (index % 4 == 2) {
			shiftx = 0.0;
			shiftz = -5.5/2.0;
		}	
		if (index % 4 == 3) {
			shiftx = -5.5/2.0;
			shiftz = 0;
		}	
	}
	if (8 <= index && index < 12) { // yz
		// shift by -x
		shiftx = -5.5;
		// array <int, 2> subs = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
		if (index % 4 == 0) {
			shifty = 0.0;
			shiftz = 5.5/2.0;
		}	
		if (index % 4 == 1) {
			shifty = 5.5/2.0;
			shiftz = 0;
		}	
		if (index % 4 == 2) {
			shifty = 0.0;
			shiftz = -5.5/2.0;
		}	
		if (index % 4 == 3) {
			shifty = -5.5/2.0;
			shiftz = 0;
		}	
	}
	// shift = segment center point
	// this is to shift radical position to the
	// center of the segment
	// imagine they are close to each other
	// then we need to substruct to center radical 
	// within the segment
	pos.x = pos.x - shiftx; // relative to its center ?
	pos.y = pos.y - shifty; // 
	pos.z = pos.z - shiftz;
	float xc, yc, zc; // rotate
	switch(type)
	{
		//Straight type
	case 1:////!!!!!the following needs to be revised and confirmed
		{xc = pos.x;
		yc = pos.y;
		zc = pos.z;
		break;}
	case 2://-z
		{xc = -pos.x;//Ry(pi)
		yc = pos.y;
		zc = -pos.z;	
		break;}
	case 3://+y
		{xc = pos.x;//Rx(pi/2)
		yc = -pos.z;
		zc = pos.y;
		break;}
	case 4:
		{xc = pos.x;
		yc = pos.z;
		zc = -pos.y;
		break;}
	case 5://+x
		{xc = -pos.z;//Ry(-pi/2)
		yc = pos.y;
		zc = pos.x;
		break;}
	case 6:
		{xc = pos.z;
		yc = pos.y;
		zc = -pos.x;
		break;}
	}
	pos.x=xc;
	pos.y=yc;
	pos.z=zc;//*/
	return pos;
}

#if RANDGEO==0
__device__ float3 pos2local(int type, float3 pos, int index)
{
//do the coordinate transformation, index is the linear index for the referred box
//from global XYZ to local XYZ so that we can use the position of DNA base in two basic type (Straight and Bend) 
	int i = index%NUCLEUS_DIM;//the x,y,z index of the box
	int j = floorf((index%(NUCLEUS_DIM*NUCLEUS_DIM))/NUCLEUS_DIM);
	int k = floorf(index/NUCLEUS_DIM/NUCLEUS_DIM);
	//printf("relative to type %d %d %d %d\n", type, x,y,z);
	// this pos is the electron position which we push into the voxel
	// say N = 67
	// x is the box index [0, 67] ... 
	// here the center of the cylinder is at zero that means it can have negative coordinates
	// but x y z can't be negative as they are just box coordantes?
	// no lol
	// box is centered around zero that's okay
	// now need to push electron into it
	// x, y, z box index that we shift into global coordinate
	// 2 * x + 1 - N = [0 N] * 2 - N -> [-N N] / 2 -> [-N / 2 ; N / 2] * UL => xvec 
	// float shiftx = (2*i + 1 - NUCLEUS_DIM)*UNITLENGTH*0.5; 
	// float shifty = (2*j + 1 - NUCLEUS_DIM)*UNITLENGTH*0.5; 
	float shiftz = (k - (NUCLEUS_DIM_Z / 2)) * UNITLENGTH + UNITLENGTH * 0.5; 
	float shifty = (j - (NUCLEUS_DIM / 2)) * UNITLENGTH + UNITLENGTH * 0.5; 
	float shiftx = (i - (NUCLEUS_DIM / 2)) * UNITLENGTH + UNITLENGTH * 0.5; 
	pos.x = pos.x - shiftx; //relative to its center ?
	pos.y = pos.y - shifty; // 
	// pos.z = pos.z-(2*z + 1 - NUCLEUS_DIM_Z)*UNITLENGTH*0.5;
	pos.z = pos.z - shiftz;
	float xc, yc, zc;
	switch(type)
	{
		//Straight type
	case 1:////!!!!!the following needs to be revised and confirmed
		{xc = pos.x;
		yc = pos.y;
		zc = pos.z;
		break;}
	case 2://-z
		{xc = -pos.x;//Ry(pi)
		yc = pos.y;
		zc = -pos.z;	
		break;}
	case 3://+y
		{xc = pos.x;//Rx(pi/2)
		yc = -pos.z;
		zc = pos.y;
		break;}
	case 4:
		{xc = pos.x;
		yc = pos.z;
		zc = -pos.y;
		break;}
	case 5://+x
		{xc = -pos.z;//Ry(-pi/2)
		yc = pos.y;
		zc = pos.x;
		break;}
	case 6:
		{xc = pos.z;
		yc = pos.y;
		zc = -pos.x;
		break;}
	case 7://Bend
		{xc = pos.x;
		yc = pos.y;
		zc = pos.z;
		break;}
	case 8:
		{xc = -pos.z;//Rz(pi)Ry(pi/2) [-Ry(pi/2)] 
		yc = -pos.y;
		zc = -pos.x;
		break;}
	case 9:
		{xc = -pos.x;//Rz(pi)
		yc = -pos.y;
		zc = pos.z;
		break;}
	case 10:
		{xc = -pos.z;//Ry(-pi/2)
		yc = pos.y;
		zc = pos.x;	
		break;}
	case 11:
		{xc = -pos.x;//Ry(pi)
		yc = pos.y;
		zc = -pos.z;
		break;}
	case 12:
		{xc = pos.z;//Rz(pi)Ry(-pi/2)
		yc = -pos.y;
		zc = pos.x;
		break;}
	case 13:
		{xc = pos.x;//Rx(pi)
		yc = -pos.y;
		zc = -pos.z;
		break;}
	case 14:
		{xc = pos.z;//Ry(pi/2)
		yc = pos.y;
		zc = -pos.x;
		break;}
	case 15:
		{xc = pos.y;//Rz(-pi/2)
		yc = -pos.x;
		zc = pos.z;
		break;}
	case 16:
		{xc = -pos.z;//Ry(-pi/2)Rz(pi/2) +
		yc = pos.x;
		zc = -pos.y;
		break;}
	case 17:
		{xc = -pos.y;//Rz(pi/2)
		yc = pos.x;
		zc = pos.z;
		break;}
	case 18:
		{xc = -pos.z;//Rz(-pi/2)Rx(pi/2)
		yc = -pos.x;
		zc = pos.y;
		break;}
	case 19:
		{xc = pos.y;//Rz(-pi/2)Ry(pi)
		yc = pos.x;
		zc = -pos.z;
		break;}
	case 20:
		{xc = pos.z;//Rz(-pi/2)Rx(-pi/2)
		yc = -pos.x;
		zc = pos.y;
		break;}
	case 21:
		{xc = -pos.y;//Rz(pi/2)Ry(pi)
		yc = -pos.x;
		zc = -pos.z;
		break;}
	case 22:
		{xc = pos.z;//Rz(pi/2)Rx(pi/2) ??
		yc = pos.x;
		zc = pos.y;
		// -y -z +x
		break;}
	case 23:
		{xc = pos.x;//Rx(pi/2)
		yc = -pos.z;
		zc = pos.y;
		break;}
	case 24:
		{xc = -pos.y;//Rz(pi/2)Ry(pi/2)
		yc = pos.z;
		zc = -pos.x;
		break;}
	case 25:
		{xc = -pos.x;//Rx(pi/2)Ry(pi) ??
		yc = pos.z;
		zc = pos.y;
		// xzz
		break;}
	case 26:
		{xc = -pos.y;//Rx(pi/2)Rz(pi/2)
		yc = -pos.z;
		zc = pos.x;
		break;}
	case 27:
		{
		xc = pos.x;//Rx(-pi/2)
		yc =pos.z;
		zc = -pos.y;	
		break;}
	case 28:
		{xc = pos.y;//Rx(pi/2)Rz(-pi/2)
		yc = -pos.z;
		zc = -pos.x;
		// -z -x y
		break;}
	case 29:
		{xc = -pos.x;//Rx(-pi/2)Ry(pi) ?
		yc = -pos.z;
		zc = -pos.y;
		break;}
	case 30:
		{xc = pos.y;//Rz(-pi/2)Ry(-pi/2)
		yc = pos.z;
		zc = pos.x;
		break;}
	default:
	    {printf("wrong type %d\n", type);  // for test
		break;}
	}
	pos.x=xc;
	pos.y=yc;
	pos.z=zc;//*/
	return pos;
}

__device__ float dist_function_sqr(float3 &a, float3 &b, float &height_up, float &height_down) {
	// two coordinates a and b
	// a = event
	// b = X-chromosome
	// if dist in x direction is radius within the dist then ok
	// if dist in y direction is diameter + 50 distance then ok
	// if dist in z direction is event - chrom (if event is lower a-b < 0 height down)
	// height down is negative
	return (abs(a.x - b.x) <= CYLINDERRADIUS) 
		&& (abs(a.y - b.y) <= 50 + CYLINDERRADIUS * 2)
		&& (height_down <= a.z - b.z) 
		&& (a.z - b.z <= height_up);
}
__device__ bool withinCylinder(float3 &a, float3 &cylinder) {
	// check height
	// a = 7204.624023 -229.854507 14569.363281 :: 7171.703613 -365.000000 14413.875977
	if ((a.z < (cylinder.z - CYLINDERHEIGHT / 2)) || 
	    ((cylinder.z + CYLINDERHEIGHT / 2) < a.z)) {
		return 0;
	}
	// check radial distance
	if ((a.x - cylinder.x) * (a.x - cylinder.x) + 
		(a.y - cylinder.y) * (a.y - cylinder.y) > 
		CYLINDERRADIUS * CYLINDERRADIUS) {
		return 0;
	}
	return 1;
}
__global__ void chemSearch(
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
	int *dev_segmentType)
{
	int id = blockIdx.x*blockDim.x+ threadIdx.x;
	curandState localState = cuseed[id%MAXNUMPAR2];
	float3 newpos, pos_cur_target;
	int3 index;
	CoorBasePair* chrom;
	float3 *histone;
	int chromNum, histoneNum,flag=0;
	while(id<num)
	{
		d_recorde[id].site.x=-1;//initialize
		d_recorde[id].site.y=-1;
		d_recorde[id].site.z=-1;
		d_recorde[id].site.w=-1;		
		d_recorde[id].prob1 = 1; //curand_uniform(&localState); // 1
		// d_recorde[id].prob1=1;
		d_recorde[id].prob2 = 0.0; // 0.6 ? 
		// threshold for prob2 ?
		
		pos_cur_target=d_edrop[id].position; // electron position / event position
		
		// ***********************************************************
		// ***********************************************************
		// ***********************************************************
		// ***********************************************************
		// here we need to modify cur position based on dev_chromosome and dev_chromosome_type
		// Step 0) skip events too far from the y=0 plane
		// if (abs(pos_cur_target.y) > 50 + CYLINDERRADIUS * 2) {
		// 	// y position is too far from the center
		// 	id+=blockDim.x*gridDim.x;
		// 	continue ;
		// }

		// Step 1) Find nearest chromosome :)
		int found_nearest_chromosome = 0;
		int id_chromosome = -1;
		for (int i = NUMCHROMOSOMES - 1; i >= 0 ; i--) { // 46
			float height_up = (dev_chromosome_type[i] / 2) * CYLINDERHEIGHT + CYLINDERHEIGHT / 2;
			float height_down = -(((dev_chromosome_type[i] - 1) / 2) * CYLINDERHEIGHT + CYLINDERHEIGHT / 2);
			// height_down is negative
			// type / 2 + 1 ~ 10 / 2 + 1 = 5
			if (dist_function_sqr(pos_cur_target, dev_chromosome[i], height_up, height_down)) {
				// FOUND NEAREST CHROMOSOME!
				// Step 1.1) Mark
				found_nearest_chromosome = 1;
				id_chromosome = i;
				break;
			}
		}
		
		if (found_nearest_chromosome == 0) {
			// Step 1.2) if for this radical we did not find
			// anything nearby, then continue to the next :) 
			id+=blockDim.x*gridDim.x;
			continue ;
		}
		
		// if we are here means we found chromosome
		// Step 1.3) Find nearest cylinder!
		int ttype = dev_chromosome_type[id_chromosome];
		int upper_part = ttype / 2;  // typy 4 :: 4 / 2 = 2 || type 5 :: 5 / 2 = 2
		int lower_part = (ttype - 1) / 2; // type 4 :: 4 / 2 - 1 = 1 || type 5 :: 5 / 2 - 1 = 1
		float3 nearest = dev_chromosome[id_chromosome];
		int found_cylinder = 0;
		int id_cylinder = -1;
		// check lower and upper parts, cylinders
		// now we need to redo cylinder ID, to be in range [1 10] instead of [0 9]
		for (int idy = 0; idy < upper_part; idy++) {
			// Step 1.4) Check left and right cylinders 
			float3 left_shift;
			left_shift.x = nearest.x + 0.0;
			left_shift.y = nearest.y - 50 - CYLINDERRADIUS; 
			left_shift.z = nearest.z + CYLINDERHEIGHT * (idy + 1);
			float3 right_shift;
			right_shift.x = nearest.x + 0.0;
			right_shift.y = nearest.y + 50 + CYLINDERRADIUS; 
			right_shift.z = nearest.z + CYLINDERHEIGHT * (idy + 1);
			if (withinCylinder(pos_cur_target, left_shift)) {
				pos_cur_target.x -= left_shift.x;
				pos_cur_target.y -= left_shift.y;
				pos_cur_target.z -= left_shift.z;
				found_cylinder = 1;
				id_chromosome = id_chromosome;
				id_cylinder = idy + lower_part + 1;
				break;
			}
			if (withinCylinder(pos_cur_target, right_shift)) {
				pos_cur_target.x -= right_shift.x;
				pos_cur_target.y -= right_shift.y;
				pos_cur_target.z -= right_shift.z;
				found_cylinder = 1;
				id_chromosome = id_chromosome + NUMCHROMOSOMES; // right side chromosome
				id_cylinder = idy + lower_part + 1;
				break;
			}
		}
		// 
		if (!found_cylinder) {
			for (int idy = 0; idy < lower_part; idy++) {
				// Step 1.4) Check left and right cylinders 
				float3 left_shift;
				left_shift.x = nearest.x + 0.0;
				left_shift.y = nearest.y - 50 - CYLINDERRADIUS; 
				left_shift.z = nearest.z - CYLINDERHEIGHT * (idy + 1); // 
				float3 right_shift;
				right_shift.x = nearest.x + 0.0;
				right_shift.y = nearest.y + 50 + CYLINDERRADIUS; 
				right_shift.z = nearest.z - CYLINDERHEIGHT * (idy + 1);
				if (withinCylinder(pos_cur_target, left_shift)) {
					pos_cur_target.x -= left_shift.x;
					pos_cur_target.y -= left_shift.y;
					pos_cur_target.z -= left_shift.z;
					found_cylinder = 1;
					id_chromosome = id_chromosome;
					id_cylinder = lower_part - 1 - idy;
					break;
				}
				if (withinCylinder(pos_cur_target, right_shift)) {
					pos_cur_target.x -= right_shift.x;
					pos_cur_target.y -= right_shift.y;
					pos_cur_target.z -= right_shift.z;
					found_cylinder = 1;
					id_chromosome = id_chromosome + NUMCHROMOSOMES; // right side chromosome
					id_cylinder = lower_part - 1 - idy;
					break;
				}		
			}
		}
		// Step 1.5) check middle part
		if (!found_cylinder) {
			// printf("Middle check\n");
			float3 left_shift;
			left_shift.x = nearest.x + 0.0;
			left_shift.y = nearest.y - CYLINDERRADIUS; 
			left_shift.z = nearest.z + 0.0;
			float3 right_shift;
			right_shift.x = nearest.x + 0.0;
			right_shift.y = nearest.y + CYLINDERRADIUS; 
			right_shift.z = nearest.z + 0.0;

			if (withinCylinder(pos_cur_target, left_shift)) {
				pos_cur_target.x -= left_shift.x;
				pos_cur_target.y -= left_shift.y;
				pos_cur_target.z -= left_shift.z;
				found_cylinder = 1;
				id_chromosome = id_chromosome; // left side chromosome
				id_cylinder = lower_part;
			}
			else
			if (withinCylinder(pos_cur_target, right_shift)) {
				pos_cur_target.x -= right_shift.x;
				pos_cur_target.y -= right_shift.y;
				pos_cur_target.z -= right_shift.z;
				found_cylinder = 1;
				id_chromosome = id_chromosome + NUMCHROMOSOMES; // right side chromosome
				id_cylinder = lower_part;
			}
		}
		// if (id < 10) {
		// 	printf("Chromosome id type :: %d %d\n", dev_chromosome_type[id_chromosome], found_cylinder);
		// }	

		if (!found_cylinder) {
			id+=blockDim.x*gridDim.x;
			continue ;
		}
		// cylinder was found and shifted appropiately
		// continue as usual
		// END OF STEP 1
		// *******************************************
		// *******************************************
		// *******************************************
		// *******************************************
		
		// from the global coordinate (-min max) to [0 N] index coordinate
		// what we know is that z must be say 6
		index.x=floorf(pos_cur_target.x/UNITLENGTH) + (NUCLEUS_DIM/2); // 2000 
		index.y=floorf(pos_cur_target.y/UNITLENGTH) + (NUCLEUS_DIM/2);
		index.z=floorf(pos_cur_target.z/UNITLENGTH) + (NUCLEUS_DIM_Z/2);
		
		// printf("It thinks Nucleosome index is %d %d %d\n", 
		// 	index.x, index.y, index.z
		// );

		int delta=index.x+index.y*NUCLEUS_DIM+index.z*NUCLEUS_DIM*NUCLEUS_DIM,minindex=-1;
		float distance[3]={100},mindis=100;
		// TO DO
		// just check 1 extra voxel nearby the wall
		flag=0;
		
		// flag changed range from 0-27 to 13-14
		for(int i=0;i<27;i++) // +6 walls
		{
			int newindex = delta+neighborindex[i];
			// if (i == 13) {
			// 	printf("ID check %d %d\n", newindex, delta);
			// }

			// flag changed Z
			if(newindex<0 || newindex > NUCLEUS_DIM*NUCLEUS_DIM*NUCLEUS_DIM_Z-1) continue;
			int type = dev_chromatinType[newindex];
			// if (i == 13) { 
			// 	printf("Type check %d\n", type);
			// }
			if(type==-1 || type==0) continue;

			newpos = pos2local(type, pos_cur_target, newindex);
			if(type<7)
			{
				chrom=dev_straightChrom;
				chromNum=STRAIGHT_BP_NUM;
				histone=dev_straightHistone;
				histoneNum=STRAIGHT_HISTONE_NUM;
			}
			else
			{
				chrom=dev_bendChrom;
				chromNum=BEND_BP_NUM;
				histone=dev_bendHistone;
				histoneNum=BEND_HISTONE_NUM;
			}
			if(flag) break;
			for(int j=0;j<chromNum;j++) // 200 nucleosome
			{
				// can take the size of base into consideration, distance should be distance-r;
				mindis=100,minindex=-1;
				distance[0] = caldistance(newpos, chrom[j].base)-RBASE;
				distance[1] = caldistance(newpos,chrom[j].left)-RSUGAR;
				distance[2] = caldistance(newpos,chrom[j].right)-RSUGAR;
				for(int iii=0;iii<3;iii++)
				{
					if(mindis>distance[iii])
					{
						mindis=distance[iii];
						minindex=iii;
					}
				}
				if(mindis<0)
				{
					if(minindex>0)
					{
						// GEANT4  
						d_recorde[id].site.x = id_chromosome; // 
						d_recorde[id].site.y = (dev_chromatinStart[newindex]+j) + TOTALBP * id_cylinder;  
						d_recorde[id].site.z = 3+minindex;
						d_recorde[id].site.w = 1; // phys or chem 0/1
					}
					flag=1;
					break;
				}
				int tmp = floorf(curand_uniform(&localState)/0.25);
				distance[0] = caldistance(newpos, chrom[j].base)-RBASE-d_rDNA[tmp];
				distance[1] = caldistance(newpos,chrom[j].left)-RSUGAR- d_rDNA[4];
				distance[2] = caldistance(newpos,chrom[j].right)-RSUGAR- d_rDNA[4];
				for(int iii=0;iii<3;iii++)
				{
					if(mindis>distance[iii])
					{
						mindis=distance[iii];
						minindex=iii;
					}
				}	
				if(mindis<0)
				{
					if(minindex>0)
					{
						// event thread id 
						d_recorde[id].site.x = id_chromosome; 
						d_recorde[id].site.y = (dev_chromatinStart[newindex]+j) + TOTALBP * id_cylinder;  
						// X-chromosome id //  
						d_recorde[id].site.z = 3+minindex; // left or right
						d_recorde[id].site.w = 1; // chem
					}
					flag=1;
					break;
				}
			}
			if(flag) break;
		}
		// Do all 6 walls * 4 each
		// CURRENT UPDATE 05/20/2022 **************************************
		// *******************************************************************
		// *******************************************************************
		// *******************************************************************
		// *******************************************************************
		// *******************************************************************
		if (flag == 0) { // still not found

			for(int i = 0; i < 24 && flag == 0; i++) // +6 walls
			{
				// printf("Checking segment number %d\n", i);
				int newdelta = delta;
				// conversions
				// the first 12 are in the current voxel so we don't need to change delta
				// 
				if (i >= 12) { // xy xz yz
					if (i < 16)  // xy +1z
						newdelta = delta + NUCLEUS_DIM*NUCLEUS_DIM;
					else if (i < 20) // xz +1y
						newdelta = delta + NUCLEUS_DIM;
					else // yz +1x
						newdelta = delta + 1;
				}
				int newindex = newdelta * 12 + i % 12;

				// ************flag changed Z
				// printf("New index vs total volume :: %d vs %d\n", newindex, NUCLEUS_DIM*NUCLEUS_DIM*NUCLEUS_DIM_Z * 12);
				if(newindex<0 || newindex >= NUCLEUS_DIM*NUCLEUS_DIM*NUCLEUS_DIM_Z * 12) continue;
				
				int type = dev_segmentType[newindex];
				if(type==-1 || type==0) continue;

				// type is not used for pos2local because we are just getting next to this cell
				// no rotation is needed here
				float3 pos_within_voxel = pos2local(1, pos_cur_target, newdelta);
				// the idea here is to shift first the position within the voxel?
				// shift relative to the voxel
				// then choose to shift next to wall center?
				// 
				newpos = PosToWall(type, pos_within_voxel, i % 12);
				// printf("Id %d and relative poistion: %0.2f %0.2f %0.2f\n", i, newpos.x, newpos.y, newpos.z);
				if(type<7)
				{
					chrom=dev_segmentChrom;
					chromNum=SEGMENT_BP_NUM;
				}
				else {
					// it's an error :)
				}
				if(flag) break;
				for(int j=0;j<chromNum;j++) // 17 SEGMENT
				{
					// can take the size of base into consideration, distance should be distance-r;
					mindis=100,minindex=-1;
					distance[0] = caldistance(newpos, chrom[j].base)-RBASE;
					distance[1] = caldistance(newpos,chrom[j].left)-RSUGAR;
					distance[2] = caldistance(newpos,chrom[j].right)-RSUGAR;
					
					for(int iii=0;iii<3;iii++)
					{
						if(mindis>distance[iii])
						{
							mindis=distance[iii];
							minindex=iii;
						}
					}
					if(mindis<0)
					{
						if(minindex>0)
						{
							// id is correct in the sense that it belongs to the 
							// event radical id
							// so we can record here anything
							// but what we need is the Chromosome ID to distinguish different DNA
							// base pair ID for damage calculations
							// and right or left dmg pair.
							d_recorde[id].site.x = id_chromosome; 
							d_recorde[id].site.y = (dev_segmentStart[newindex]+j) + TOTALBP * id_cylinder;  
							d_recorde[id].site.z = 3+minindex;
							d_recorde[id].site.w = 1; // phys or chem
						}
						flag=1; // found
						break;
					}
					int tmp = floorf(curand_uniform(&localState)/0.25);
					distance[0] = caldistance(newpos, chrom[j].base)-RBASE-d_rDNA[tmp];
					distance[1] = caldistance(newpos,chrom[j].left)-RSUGAR- d_rDNA[4];
					distance[2] = caldistance(newpos,chrom[j].right)-RSUGAR- d_rDNA[4];
					for(int iii=0;iii<3;iii++)
					{
						if(mindis>distance[iii])
						{
							mindis=distance[iii];
							minindex=iii;
						}
					}	
					if(mindis<0)
					{
						if(minindex>0)
						{
							d_recorde[id].site.x = id_chromosome; 
							d_recorde[id].site.y = (dev_segmentStart[newindex]+j) + TOTALBP * id_cylinder;  
							d_recorde[id].site.z = 3+minindex;
							d_recorde[id].site.w = 1; // phys or chem
						}
						flag=1;
						break;
					}
				}
				if(flag) break;
			}
		}
		id+=blockDim.x*gridDim.x;
	}
	cuseed[id%MAXNUMPAR2]=localState;
}

__global__ void phySearch(
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
	int *dev_segmentType)
{
	int id = blockIdx.x*blockDim.x+ threadIdx.x;
	curandState localState = cuseed[id%MAXNUMPAR2];
	float3 newpos, pos_cur_target;
	int3 index;
	CoorBasePair* chrom;
	float3 *histone;
	int chromNum, histoneNum,flag=0;
	while(id<num)
	{
		d_recorde[id].site.x=-1;//initialize
		d_recorde[id].site.y=-1;
		d_recorde[id].site.z=-1;
		d_recorde[id].site.w=-1;		
		d_recorde[id].prob1=d_edrop[id].e;
		d_recorde[id].prob2=0.0; //curand_uniform(&localState)*(EMAX-EMIN) + EMIN; // constant 
		// threshold for prob2

		pos_cur_target=d_edrop[id].position;
		// ***********************************************************
		// ***********************************************************
		// ***********************************************************
		// ***********************************************************
		// here we need to modify cur position based on dev_chromosome and dev_chromosome_type
		// Step 0) skip events too far from the y=0 plane
		// if (id < num) {
		// 	printf("Physical search %d %d\n", id, num);
		// }

		// if (abs(pos_cur_target.y) > 50 + CYLINDERRADIUS * 2) {
		// 	// y position is too far from the center
		// 	id+=blockDim.x*gridDim.x;
		// 	continue ;
		// }
		// printf("Within the plane! %d\n", id);
		// Step 1) Find nearest chromosome :)
		bool found_nearest_chromosome = 0;
		int id_chromosome = -1;
		for (int i = 0; i < NUMCHROMOSOMES; i++) {
			float height_up = (dev_chromosome_type[i] / 2) * CYLINDERHEIGHT + CYLINDERHEIGHT / 2;
			float height_down = -(((dev_chromosome_type[i] - 1) / 2) * CYLINDERHEIGHT + CYLINDERHEIGHT / 2);
			if (dist_function_sqr(pos_cur_target, dev_chromosome[i], height_up, height_down)) {
				// FOUND NEAREST CHROMOSOME!
				// Step 1.1) Mark
				found_nearest_chromosome = 1;
				id_chromosome = i;
				break;
			}
		}
		if (found_nearest_chromosome == 0) {
			// Step 1.2) if for this radical we did not find
			// anything nearby, then continue to the next :) 
			id+=blockDim.x*gridDim.x;
			continue ;
		}
		// printf("Near some Chromosome! %d %d\n", id, id_chromosome);
		// if we are here means we found chromosome
		// Step 1.3) Find nearest cylinder!
		int ttype = dev_chromosome_type[id_chromosome];
		int upper_part = ttype / 2;  // typy 4 :: 4 / 2 = 2 || type 5 :: 5 / 2 = 2
		int lower_part = (ttype - 1) / 2; // type 4 :: (4 - 1) / 2 = 1 || type 5 :: (5 - 1) / 2 = 2
		float3 nearest = dev_chromosome[id_chromosome];
		// printf("Cur pos and chromosome %f %f %f && %f %f %f\n", 
		// 	pos_cur_target.x, pos_cur_target.y, pos_cur_target.z, 
		// 	nearest.x, nearest.y, nearest.z);
		bool found_cylinder = 0;
		int id_cylinder = -1;
		// check lower and upper parts, cylinders
		for (int idy = 0; idy < upper_part; idy++) {
			// Step 1.4) Check left and right cylinders 
			float3 left_shift;
			left_shift.x = nearest.x + 0.0;
			left_shift.y = nearest.y - 50 - CYLINDERRADIUS; 
			left_shift.z = nearest.z + CYLINDERHEIGHT * (idy + 1);
			float3 right_shift;
			right_shift.x = nearest.x + 0.0;
			right_shift.y = nearest.y + 50 + CYLINDERRADIUS; 
			right_shift.z = nearest.z + CYLINDERHEIGHT * (idy + 1);
			if (withinCylinder(pos_cur_target, left_shift)) {
				pos_cur_target.x -= left_shift.x;
				pos_cur_target.y -= left_shift.y;
				pos_cur_target.z -= left_shift.z;
				found_cylinder = 1;
				id_chromosome = id_chromosome;
				id_cylinder = idy + lower_part + 1;
				break;
			}
			if (withinCylinder(pos_cur_target, right_shift)) {
				pos_cur_target.x -= right_shift.x;
				pos_cur_target.y -= right_shift.y;
				pos_cur_target.z -= right_shift.z;
				found_cylinder = 1;
				id_chromosome = id_chromosome + NUMCHROMOSOMES; // right side chromosome
				id_cylinder = idy + lower_part + 1;
				break;
			}
		}
		// 
		if (!found_cylinder) {
			for (int idy = 0; idy < lower_part; idy++) {
				// Step 1.4) Check left and right cylinders 
				float3 left_shift;
				left_shift.x = nearest.x + 0.0;
				left_shift.y = nearest.y - 50 - CYLINDERRADIUS; 
				left_shift.z = nearest.z - CYLINDERHEIGHT * (idy + 1);
				float3 right_shift;
				right_shift.x = nearest.x + 0.0;
				right_shift.y = nearest.y + 50 + CYLINDERRADIUS; 
				right_shift.z = nearest.z - CYLINDERHEIGHT * (idy + 1);
				if (withinCylinder(pos_cur_target, left_shift)) {
					pos_cur_target.x -= left_shift.x;
					pos_cur_target.y -= left_shift.y;
					pos_cur_target.z -= left_shift.z;
					found_cylinder = 1;
					id_chromosome = id_chromosome;
					id_cylinder = lower_part - 1 - idy;
					break;
				}
				if (withinCylinder(pos_cur_target, right_shift)) {
					pos_cur_target.x -= right_shift.x;
					pos_cur_target.y -= right_shift.y;
					pos_cur_target.z -= right_shift.z;
					found_cylinder = 1;
					id_chromosome = id_chromosome + NUMCHROMOSOMES; // right side chromosome
					id_cylinder = lower_part - 1 - idy;
					break;
				}		
			}
		}
		// Step 1.5) check middle part
		if (!found_cylinder) {
			float3 left_shift;
			left_shift.x = nearest.x + 0.0;
			left_shift.y = nearest.y - CYLINDERRADIUS; 
			left_shift.z = nearest.z + 0.0;
			float3 right_shift;
			right_shift.x = nearest.x + 0.0;
			right_shift.y = nearest.y + CYLINDERRADIUS; 
			right_shift.z = nearest.z + 0.0;
			if (withinCylinder(pos_cur_target, left_shift)) {
				pos_cur_target.x -= left_shift.x;
				pos_cur_target.y -= left_shift.y;
				pos_cur_target.z -= left_shift.z;
				found_cylinder = 1;
				id_chromosome = id_chromosome; // left side chromosome
				id_cylinder = lower_part;
			}
			else
			if (withinCylinder(pos_cur_target, right_shift)) {
				pos_cur_target.x -= right_shift.x;
				pos_cur_target.y -= right_shift.y;
				pos_cur_target.z -= right_shift.z;
				found_cylinder = 1;
				id_chromosome = id_chromosome + NUMCHROMOSOMES; // right side chromosome
				id_cylinder = lower_part;
			}
		}

		if (!found_cylinder) {
			id+=blockDim.x*gridDim.x; // event id damage deposition id
			continue ;
		}
		// printf("Within some cylinder! %d\n", id);
		// cylinder was found and shifted appropiately
		// continue as usual
		// END OF STEP 1
		// *******************************************
		// *******************************************
		// *******************************************
		// *******************************************
		
		index.x=floorf(pos_cur_target.x/UNITLENGTH) + (NUCLEUS_DIM/2); // 2000 
		index.y=floorf(pos_cur_target.y/UNITLENGTH) + (NUCLEUS_DIM/2);
		index.z=floorf(pos_cur_target.z/UNITLENGTH) + (NUCLEUS_DIM_Z/2);
		// printf("It thinks Nucleosome index is %d %d %d\n", 
		// 	index.x, index.y, index.z
		// );
		int delta=index.x+index.y*NUCLEUS_DIM+index.z*NUCLEUS_DIM*NUCLEUS_DIM,minindex=-1;
		float distance[3]={100},mindis=100;
		for(int i=0;i<27;i++)
		{
			flag=0;
			int newindex = delta+neighborindex[i];
			if(newindex<0 || newindex > NUCLEUS_DIM*NUCLEUS_DIM*NUCLEUS_DIM_Z-1) continue;
			int type = dev_chromatinType[newindex];
			if(type==-1 || type==0) continue;

			newpos = pos2local(type, pos_cur_target, newindex);
			// if (id < 10) {
				// printf("type = %d\n", type);
				// printf("local pos %f %f %f\n", newpos.x, newpos.y, newpos.z);
			// }
			if(type<7)
			{
				chrom=dev_straightChrom;
				chromNum=STRAIGHT_BP_NUM;
				histone=dev_straightHistone;
				histoneNum=STRAIGHT_HISTONE_NUM;
			}
			else
			{
				chrom=dev_bendChrom;
				chromNum=BEND_BP_NUM;
				histone=dev_bendHistone;
				histoneNum=BEND_HISTONE_NUM;
			}
			// for(int j=0;j<histoneNum;j++)
			// {
			// 	mindis = caldistance(newpos, histone[j]) - RHISTONE;
			// 	if(mindis < 0) flag=1;
			// }
			// printf("flag lol %d\n", flag);
			if(flag) break;
			// printf("Avoided flag\n");
			for(int j=0;j<chromNum;j++) // 200 // nucleosome
			{
				// can take the size of base into consideration, distance should be distance-r;
				mindis=100,minindex=-1;
				distance[0] = caldistance(newpos, chrom[j].base)-RBASE-RPHYS;
				distance[1] = caldistance(newpos,chrom[j].left)-RSUGAR- RPHYS;
				distance[2] = caldistance(newpos,chrom[j].right)-RSUGAR- RPHYS;
				for(int iii=0;iii<3;iii++)
				{
					if(mindis>distance[iii])
					{
						mindis=distance[iii];
						minindex=iii;
					}
				}
				// 
				// if (mindis < 1.0) {
				// 	printf("mindis %f  and bp_id = %d\n", mindis, j);
				// }
				if(mindis<0)
				{
					// printf("found mindis %f\n", mindis);
					if(minindex>0)
					{
						//[10 8 7 6 6 .... ] 
						// [0 1 2 3 4] index of a X-chromosome
						// [10 19 26 ... ] cylinder index
						// 120000 * 200 * 522 cylinders 12,000,000,000
						// we don't need site.x previous definition
						// we do need to add cylinder ID and chromosome
						// printf("found x\n");
						// 26813034
						d_recorde[id].site.x = id_chromosome; 
						d_recorde[id].site.y = (dev_chromatinStart[newindex]+j) + TOTALBP * id_cylinder;  
						d_recorde[id].site.z = 3+minindex;
						d_recorde[id].site.w = 0;
						if (d_recorde[id].site.y == 8290192) {
							printf("Voxel Found!\n");
							printf("Related chromosome type :: %d\n", dev_chromosome_type[id_chromosome]);
							printf("Related chromosome ID :: %d\n", id_chromosome);
							printf("Related cylinder ID :: %d\n", id_cylinder);
							printf("Related index :: %d %d %d\n", index.x, index.y, index.z);
							printf("voxel related pos original :: %f %f %f\n", 
								d_edrop[id].position.x, 
								d_edrop[id].position.y, 
								d_edrop[id].position.z);
							printf("voxel related phy energy :: %f", d_edrop[id].e);

						}
					}
					flag=1;
				}
			}
			if(flag) break;
		}
		// Do all 6 walls * 4 each
		// CURRENT UPDATE 05/20/2022 **************************************
		// *******************************************************************
		// *******************************************************************
		// *******************************************************************
		// *******************************************************************
		// *******************************************************************
		if (flag == 0) { // still not found

			for(int i = 0; i < 24 && flag == 0; i++) // +6 walls
			{
				// printf("Checking segment number %d\n", i);
				int newdelta = delta;
				// conversions
				// the first 12 are in the current voxel so we don't need to change delta
				// 
				if (i >= 12) { // xy xz yz
					if (i < 16)  // xy +1z
						newdelta = delta + NUCLEUS_DIM*NUCLEUS_DIM;
					else if (i < 20) // xz +1y
						newdelta = delta + NUCLEUS_DIM;
					else // yz +1x
						newdelta = delta + 1;
				}
				int newindex = newdelta * 12 + i % 12;

				// ************flag changed Z
				// printf("New index vs total volume :: %d vs %d\n", newindex, NUCLEUS_DIM*NUCLEUS_DIM*NUCLEUS_DIM_Z * 12);
				if(newindex<0 || newindex >= NUCLEUS_DIM*NUCLEUS_DIM*NUCLEUS_DIM_Z * 12) continue;
				
				int type = dev_segmentType[newindex];
				if(type==-1 || type==0) continue;

				float3 pos_within_voxel = pos2local(1, pos_cur_target, newdelta);
				newpos = PosToWall(type, pos_within_voxel, i % 12);
				if(type<7)
				{
					chrom=dev_segmentChrom;
					chromNum=SEGMENT_BP_NUM;
				}
				else {
					// it's an error :)
				}
				if(flag) break;
				for(int j=0;j<chromNum;j++) // 17 SEGMENT
				{
					// can take the size of base into consideration, distance should be distance-r;
					mindis=100,minindex=-1;
					distance[0] = caldistance(newpos, chrom[j].base) - RBASE - RPHYS;
					distance[1] = caldistance(newpos, chrom[j].left) - RSUGAR - RPHYS;
					distance[2] = caldistance(newpos, chrom[j].right) - RSUGAR - RPHYS;
					
					for(int iii=0;iii<3;iii++)
					{
						if(mindis>distance[iii])
						{
							mindis=distance[iii];
							minindex=iii;
						}
					}
					if(mindis<0)
					{
						if(minindex>0)
						{
							// id is correct in the sense that it belongs to the 
							// event radical id
							// so we can record here anything
							// but what we need is the Chromosome ID to distinguish different DNA
							// base pair ID for damage calculations
							// and right or left dmg pair.
							// printf("Found something xD\n");
							d_recorde[id].site.x = id_chromosome;  // 92 ? 
							d_recorde[id].site.y = (dev_segmentStart[newindex]+j) + TOTALBP * id_cylinder;  
							d_recorde[id].site.z = 3+minindex; // left or right pair
							d_recorde[id].site.w = 0; // phys or chem
						}
						flag=1; // found
						break;
					}
				}
				if(flag) break;
			}
		}	
		//if(id%(blockDim.x*gridDim.x)==0) printf("id is %d\n", id);
		id+=blockDim.x*gridDim.x;//*/
	}
	cuseed[id%MAXNUMPAR2]=localState;
}//*/
#endif
/***********************************************************************************/

Edeposit* readStage(int *numPhy, int mode, int file_id)
/*******************************************************************
c*    Reads electron reactive events from physics stage result     *
c*    Setup electron events as a list for the DNA damages          *
output *effphy 
Number of effective Physics damage
c******************************************************************/
{
	int start,stop;
	float data[4];
	Edeposit *hs = NULL;
	int len = 0, prev_len = 0;
	{
		// cout << file_id << " ";
		ifstream infile;
		if(mode==0) {
			string input = REALTIME_FILEIN + to_string(file_id) + "/totalphy.dat"; 
			infile.open(input,ios::binary);
			// printf("physics results: Reading %s\n", input.c_str());
		}	
		else {
			string input = REALTIME_FILEIN + to_string(file_id) + FILEOH + ".dat";
			infile.open(input,ios::binary);
			// printf("chemistry results: Reading %s\n", input.c_str());
		}
		start=infile.tellg();
		infile.seekg(0, ios::end);
		stop=infile.tellg();
		len=(stop-start)/16;
		if(len==0) { infile.close(); return hs; }
		infile.seekg(0, ios::beg);
		hs = (Edeposit *)malloc(sizeof(Edeposit)*(prev_len + len));
		
		for(int j=prev_len;j<prev_len + len;j++)
		{
			infile.read(reinterpret_cast <char*> (&data), sizeof(data));
			hs[j].position.x=data[0];
			hs[j].position.y=data[1];
			hs[j].position.z=data[2];
			if(mode==0) hs[j].e=data[3];
			else hs[j].e=1-PROBCHEM;
		} 
		prev_len += len;
		infile.close();
	} 
	// cout << endl;
	(*numPhy) += prev_len;
 	return hs;
}

void quicksort(chemReact*  hits,int start, int stop, int sorttype)
{   
    //CPU sort function for ordering chemReacts in cpu memory
    switch(sorttype)
    {
	    case 1:
	    {   sort(hits+start,hits+stop,compare1);
	        break;
	    }
	    case 2:
	    {   sort(hits+start,hits+stop,compare2);
	        break;
	    }
	    default:
	    {   sort(hits+start,hits+stop,compare1);
	        break;
	    }
    }
}
chemReact* combinePhy(int* totalphy, combinePhysics* recorde, int mode, int file_id)
{
	int counts=(*totalphy);
	sort(recorde,recorde+counts,compare3);
	
	int j,num=0;
	// printf("CombinePhy counts %d\n", counts);
    for(int i=0; i<counts;)
    {
		if (recorde[i].site.z==-1) {i++;continue;}
    	j=i+1;
        while(recorde[j].site.x==recorde[i].site.x) // id base pair
        {
        	if(recorde[j].site.y==recorde[i].site.y && recorde[j].site.z==recorde[i].site.z)
        	{
        		if (mode == 0) recorde[i].prob1 +=recorde[j].prob1; // sum up energies
        		else recorde[i].prob2 *= recorde[j].prob2; // mode 1 for chem stage 
				// for mode 1: reduction of the threshold
        		recorde[j].site.z=-1;
        	}
        	j++;
        	if(j==counts) break;
        }        	
        i++;
    }
	// why are we doing these counts?
    for(int i=0;i<counts;i++)
    {
		if(recorde[i].site.z!=-1 && recorde[i].prob2<recorde[i].prob1)
    	{
    		num++;
    	}
    }
	// printf("counts after probabilities %d\n", num);
    if(num==0) {(*totalphy)=0;return NULL;}
	string output = REALTIME_FILEOUT + to_string(file_id) + "_" + FILEOHNAME + ".txt";
	cout << output << endl;

	ofstream fout;
	if (mode == 0) { // at first open and replace
		fout.open(output.c_str());
	}
	else { // then append
		fout.open(output.c_str(), std::ios_base::app);
	}
	
    chemReact* recordPhy=(chemReact*) malloc(sizeof(chemReact)*num);
    int index=0;
    for(int i=0;i<counts;i++)
    {
		if (recorde[i].site.z != -1) { // if found event near to base pair
			fout << mode << " " << // mode  //0 
				recorde[i].site.x << " " <<  // chrom id // 1
				recorde[i].site.y << " " <<  // bp id // 2
				recorde[i].site.z << " " <<  // left/right // 3
				recorde[i].site.w << " " <<  // 1/0 phy/chem // 5
				recorde[i].prob1 << endl; // energy or prob 0.6
		}
    	if(recorde[i].site.z!=-1 && recorde[i].prob2<recorde[i].prob1)
    	{
    		recordPhy[index].x=recorde[i].site.x;
    		recordPhy[index].y=recorde[i].site.y;
    		recordPhy[index].z=recorde[i].site.z;
    		recordPhy[index].w=recorde[i].site.w;
    		index++;
    	}
    }
    (*totalphy)=num;
    return recordPhy;
}
void damageAnalysis(int counts, chemReact* recordpos, int numFiles, float dose)
{
	// seems currently only the number of total SSB or DSB are correct
	// be careful to use the number in each category!!
	if(counts==0) return;
	char buffer[256];
	int complexity[7]={0};//SSB, 2xSSB, SSB+, 2SSB, DSB, DSB+, DSB++
	int results[7]={0}; //SSBd,  SSbi, SSbm, DSBd, DSBi, DSBm, DSBh.
	
	quicksort(recordpos,0,counts,1);
	// sort, this sorts x first which id DNA id, then it sorts by bp id
    int start=0,m,numofstrand,numoftype,k,cur_dsb;
	// printf("counts %d\n", counts);
    for(int i=0; i<counts;) // go over all one by one?
    {
    	if(recordpos[i].z==-1) {i++;continue;} // skip if it's not damaged? why recorded lol
    	start=i;
        while(i<counts)
        {
        	if(recordpos[i].x==recordpos[start].x) i++; // if it's the same DNA ?
        	else break;
        }
		// printf("Ids of the record and next? :: %d %d\n", start, i);
        if(i==start+1)//only one break on the DNA whole DNA, rather it's the end of the DNA ... 
        {
        	complexity[0]++; // single break
        	results[recordpos[start].w]++;
        	continue;//find breaks in another DNA
        }
		// range [start -> i] same DNA 
        if(i>start+1) quicksort(recordpos,start,i,2);//order damage sites so that search can be done ?
		cur_dsb=0;
        for(k=start;k<i-1;)//more than one break range [k -> i) of the DNA
        {
        	if(recordpos[k+1].y-recordpos[k].y>dS)
        	{
        		complexity[1]++; // 2xSSB
        		results[recordpos[k].w]++;
        		k++;
        		continue;
        	}
        	else
        	{
	        	m=k+1;
	        	numoftype=0;
	        	numofstrand=0;
	        	int flag=0;//means SSB, 1 for DSB
        		while(m<i)
        		{
        			if( recordpos[m].z!=recordpos[m-1].z)//recordpos[m].y-recordpos[m-1].y<dDSB &&
        			{ // left + right
        				numofstrand++;
        				if(recordpos[m].w!=recordpos[k].w) numoftype++; // phys+chem
        				int j=m; // next
        				int tmptype=0;
        				for(;j>k-1;j--) // go back?
        				{
        					if(recordpos[m].y-recordpos[j].y>dDSB) break; // more than 10 
        					if(recordpos[j].w!=recordpos[k].w) tmptype++; // phys+chem
        				}

        				if(j==k-1) flag=1;//DSB k->m less then 10 all
        				else if(j==k && m==k+1) flag=2;//2SSB didn't reach k-> m > 10
        				else {m=j+1;numoftype-=tmptype;}
        				break; // end here if left + right
        			}
        			if(recordpos[m].y-recordpos[k].y>dS) {m--;break;}//SSB+
        			if(recordpos[m].w!=recordpos[k].w) numoftype++;
    				m++;
        		}
        		if(flag==0) // SSB ? 
        		{
        			complexity[2]++; // SSB+ ?
	        	 	if(numoftype!=0) results[2]++; 
	        		else results[recordpos[k].w]++;//=m-k;
        		}
        		else if(flag==2)
        		{
        			complexity[3]++; // 2SSB
	        	 	if(numoftype!=0) results[2]++;
	        		else results[recordpos[k].w]++;
        		}
	        	else
	        	{//if flag=1,m must be k+1 and from k there must be a DSB
	        		m=k;//in consitent with the calculation of chem type,
	        		numoftype=0;
	        		int numofchem=0;
	        		while(m<i)
	        		{
	        			if(recordpos[m].y-recordpos[k].y<dDSB)
	        			{
	        				if(recordpos[m].w!=recordpos[k].w) numoftype++;
	        				if(recordpos[m].w==1) numofchem++;
	        				m++;
	        			}
	        			else
	        				break;
	        		}
	        		if(numofchem==1) results[6]++;
	        		else if(numoftype!=0) results[5]++;
	        		else results[3+recordpos[k].w]++;

	        		if(m-k==2) complexity[4]++; // DSB
	        		else complexity[5]++; // DSB+
	        		cur_dsb++;
	        	}
	        	k=m;
        	}       	
        }
        if(cur_dsb>1) complexity[6]++; // DSB++
        if(k==i-1)//deal with the last one in a segment
        {
        	complexity[1]++;
        	results[recordpos[k].w]++;
        }
    }

    FILE* fp = fopen("./Results/finalstat.txt","a");
	int ssbs = 0, dsbs = 0;
	for (int i = 0; i < 3; i++) ssbs += results[i];
	for (int i = 3; i < 7; i++) dsbs += results[i];

	fprintf(fp, "%d %d %d %f\n", numFiles, ssbs, dsbs, dose);
	printf("%d %d %d %f\n", numFiles, ssbs, dsbs, dose);
    
	fprintf(fp, "SSBd SSbi SSbm DSBd DSBi DSBm DSBh\n");
    for(int index = 0; index < 7; index++)
    	fprintf(fp, "%d ", results[index]);
    fprintf(fp, "\n");
    fprintf(fp, "SSB 2xSSB SSB+ 2SSB DSB DSB+ DSB++\n");
    for(int index = 0; index < 7; index++)
    	fprintf(fp, "%d ", complexity[index]);
   	fprintf(fp, "\n");
	fclose(fp);//*/
}

#endif
