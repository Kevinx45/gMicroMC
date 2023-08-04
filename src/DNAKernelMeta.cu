#include "DNAKernelMeta.cuh"
#include "DNAKernel.cu"
void DNAList::initDNAMeta()
{
  int totalspace = NUCLEUS_DIM_META*NUCLEUS_DIM_META*NUCLEUS_DIM_Z_META;
		int *chromatinIndex = (int*)malloc(sizeof(int)*totalspace);
		int *chromatinStart = (int*)malloc(sizeof(int)*totalspace);
		int *chromatinType = (int*)malloc(sizeof(int)*totalspace);
    
		for (int k=0; k<totalspace; k++) 
		{
			chromatinIndex[k] = -1;
			chromatinStart[k] = -1;
			chromatinType[k] = -1;
		}
    
		int totalspace_sub = NUCLEUS_DIM_META*NUCLEUS_DIM_META*NUCLEUS_DIM_Z_META * 12;
		int *segmentIndex = (int*)malloc(sizeof(int)*totalspace_sub);
		int *segmentStart = (int*)malloc(sizeof(int)*totalspace_sub);
		int *segmentType = (int*)malloc(sizeof(int)*totalspace_sub);
		for (int k=0; k<totalspace_sub; k++) 
		{
			segmentIndex[k] = -1;
			segmentStart[k] = -1;
			segmentType[k] = -1;
		}


    // X-CHROMOSOMES, there are 46 of them 
    int data[6];
		std::cout << "Reading the chromosomes and types?\n";
		std::ifstream fin;
		fin.open(document["chromCoords"].GetString()); // v5 has 0,0,0 chromosome
		float fdata[3];
		// CoorBasePair *StraightChrom = (CoorBasePair*)malloc(sizeof(CoorBasePair)*STRAIGHT_BP_NUM);
		float3 *chromosome = (float3*)malloc(sizeof(float3) * NUMCHROMOSOMES_META);
		int *chromosome_type = (int*)malloc(sizeof(int) * NUMCHROMOSOMES_META);
		float ttype;
		for (int i = 0; fin >> fdata[0] >> fdata[1] >> fdata[2] >> ttype; i++) { // 46 x-chromosomes
			chromosome[i].x = fdata[0];
			chromosome[i].y = fdata[1];
			chromosome[i].z = fdata[2];
			chromosome_type[i] = ttype;
			if (i < 5) printf("%f %f %f %d\n", fdata[0], fdata[1], fdata[2], chromosome_type[i]);
		}
		fin.close();


	
		// long lSize;
		// FILE* pFile=fopen("./table/WholeNucleoChromosomesTable.bin","rb");
		// fseek (pFile , 0 , SEEK_END);
	    // lSize = ftell (pFile);
	  	// rewind (pFile);
	  	// for (int i=0; i<lSize/(6*sizeof(int)); i++)
		// {
		//     fread(data,sizeof(int),6, pFile);
		//     //if(i<5) printf("%d %d %d %d %d %d\n", data[0], data[1], data[2], data[3], data[4], data[5]);
		// 	index = data[0] + data[1] * NUCLEUS_DIM + data[2] * NUCLEUS_DIM * NUCLEUS_DIM;
		// 	chromatinIndex[index] = data[3];
		// 	chromatinStart[index] = data[4];
		// 	chromatinType[index] = data[5];
		// }
		// fclose(pFile);
		
		
		CUDA_CALL(cudaMalloc((void**)&dev_chromosome, NUMCHROMOSOMES_META * sizeof(float3)));
		CUDA_CALL(cudaMemcpy(dev_chromosome, chromosome, NUMCHROMOSOMES_META * sizeof(float3), cudaMemcpyHostToDevice));

		CUDA_CALL(cudaMalloc((void**)&dev_chromosome_type, NUMCHROMOSOMES_META * sizeof(int)));
		CUDA_CALL(cudaMemcpy(dev_chromosome_type, chromosome_type, NUMCHROMOSOMES_META * sizeof(int), cudaMemcpyHostToDevice));
		
		std::cout << "Time to read voxelized coordinates \n";
		//ifstream fin;
		// I need to figure out how to get extra coordinates
		// maybe I can store in the chromatin index as I'm not using it anyway
		fin.open(document["voxelizedCoords"].GetString());
		// ./Results/voxelized_coordinates_b_v4_connected.txt
		for (int i=0; fin >> data[0] >> data[1] >> data[2] >> data[3] >> data[4] >> data[5]; i++)
		{
			//fread(data,sizeof(int),6, pFile);
			if(i<5) printf("%d %d %d %d %d %d\n", data[0], data[1], data[2], data[3], data[4], data[5]);
			// first 3 are indicies
			if (data[3] == 0) {
				int index = data[0] + data[1] * NUCLEUS_DIM_META + data[2] * NUCLEUS_DIM_META * NUCLEUS_DIM_META;
				chromatinIndex[index] = data[3]; // index of the extra nucleosome ?
				chromatinStart[index] = data[4]; // bp index 200
				chromatinType[index] = data[5]; // type
			}
			else {
				// Step 1)
				// convert to voxel id first
				int x = data[0]; // segment sub voxel ids
				int y = data[1];
				int z = data[2];
				int xx = x / 4; // center voxel id
				int yy = y / 4;
				int zz = z / 4;
				int xxx = x % 4; // subvoxel coordinates
				int yyy = y % 4;
				int zzz = z % 4;
				// Step 2) Convert using 'convention'
				// convention :: we have 3 walls with 4 subvoxels each
				// numerated clockwise
				// walls xy xz yz
				// subvoxels [(0, 1), (1, 0), (0, -1), (-1, 0)]
				std::array <int, 2> subs[4] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
				// ids [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
				int subvoxel_id = 0;
				xxx -= 2; // center subvoxel
				yyy -= 2;
				zzz -= 2;
				std::array <int, 2> nn;
				if (zzz == -2) { // this is xy plane
					subvoxel_id += 0;
					nn = {xxx, yyy};
				}
				if (yyy == -2) { // xz
					subvoxel_id += 4;
					nn = {xxx, zzz};
				}
				if (xxx == -2) { // yz plane
					subvoxel_id += 8;
					nn = {yyy, zzz};
				}
				for (int j = 0; j < 4; j++) {
					if (nn == subs[j]) {
						subvoxel_id += j;
						break ;
					}
				}
				// index = x + y * NUCLEUS_DIM + z * NUCLEUS_DIM * NUCLEUS_DIM; // current id of the voxel
				int index = xx + yy * NUCLEUS_DIM_META + zz * NUCLEUS_DIM_META * NUCLEUS_DIM_META;
				int sub_index = index * 12; // shifted index to accommodate 12 subvoxels
				sub_index += subvoxel_id;
				// [544.5, 170.5, 93.5]
				// if ((float)xx * 11 + 5.5 == 544.5 && 
				// 	(float)yy * 11 + 5.5 == 170.5 &&
				// 	(float)zz * 11 + 5.5 == 93.5) {
				// 	cout << "index :: " << index << " " << xx << " " << yy << " " << zz << endl;
				// 	cout << "ID of a segment and type = " << sub_index << " " << data[3] << " " << data[4] << " " << data[5] << endl;
				// }
				segmentIndex[sub_index] = data[3]; // future chromosome ID
				segmentStart[sub_index] = data[4]; // segment base pair start
				segmentType[sub_index] = data[5]; // type orientation
			}
		}
		fin.close();
		std::cout << "end of reading voxelized coordinates \n\n";
		CUDA_CALL(cudaMalloc((void**)&dev_chromatinIndex, totalspace * sizeof(int)));
		CUDA_CALL(cudaMemcpy(dev_chromatinIndex, chromatinIndex, totalspace * sizeof(int), cudaMemcpyHostToDevice));//DNA index
		CUDA_CALL(cudaMalloc((void**)&dev_chromatinStart, totalspace * sizeof(int)));
		CUDA_CALL(cudaMemcpy(dev_chromatinStart, chromatinStart, totalspace * sizeof(int), cudaMemcpyHostToDevice));//# of start base in the box
		CUDA_CALL(cudaMalloc((void**)&dev_chromatinType, totalspace * sizeof(int)));
		CUDA_CALL(cudaMemcpy(dev_chromatinType, chromatinType, totalspace * sizeof(int), cudaMemcpyHostToDevice));//type of the DNA in the box
	    free(chromatinIndex);
	    free(chromatinStart);
	    free(chromatinType);
		// copying all segments into CUDA
		CUDA_CALL(cudaMalloc((void**)&dev_segmentIndex, totalspace_sub * sizeof(int)));
		CUDA_CALL(cudaMemcpy(dev_segmentIndex, segmentIndex, totalspace_sub * sizeof(int), cudaMemcpyHostToDevice));//DNA index
		CUDA_CALL(cudaMalloc((void**)&dev_segmentStart, totalspace_sub * sizeof(int)));
		CUDA_CALL(cudaMemcpy(dev_segmentStart, segmentStart, totalspace_sub * sizeof(int), cudaMemcpyHostToDevice));//# of start base in the box
		CUDA_CALL(cudaMalloc((void**)&dev_segmentType, totalspace_sub * sizeof(int)));
		CUDA_CALL(cudaMemcpy(dev_segmentType, segmentType, totalspace_sub * sizeof(int), cudaMemcpyHostToDevice));//type of the DNA in the box
	    free(segmentIndex);
	    free(segmentStart);
	    free(segmentType);
		// end copying segments
		// Loading segment template
		CoorBasePair *SegmentChrom = (CoorBasePair*)malloc(sizeof(CoorBasePair)*SEGMENT_BP_NUM_META);
		const char *segment = document["segmentChromatinMeta"].GetString();
		printf("Segment Chromatin Table: Reading %s\n", segment);
		FILE *fpSegment = fopen(segment,"r");
		float dump_float;
    	int dump;
		float bx, by, bz, rx, ry, rz, lx, ly, lz;
	    for (int i=0; i<SEGMENT_BP_NUM_META; i++)
		{
		    fscanf(fpSegment,"%f %f %f %f %f %f %f %f %f %f\n", &dump_float, &bx, &by, &bz, &rx, &ry, &rz, &lx, &ly, &lz);
			dump = dump_float;
			if(i<5) printf("%d %f %f %f %f %f %f %f %f %f\n", dump, bx, by, bz, rx, ry, rz, lx, ly, lz);
			SegmentChrom[i].base.x = bx;
			SegmentChrom[i].base.y = by;
			SegmentChrom[i].base.z = bz;
			SegmentChrom[i].right.x = rx;
			SegmentChrom[i].right.y = ry;
			SegmentChrom[i].right.z = rz;
			SegmentChrom[i].left.x = lx;
			SegmentChrom[i].left.y = ly;
			SegmentChrom[i].left.z = lz;
		}
		fclose(fpSegment);
		CUDA_CALL(cudaMalloc((void**)&dev_segmentChrom, SEGMENT_BP_NUM_META * sizeof(CoorBasePair)));
		CUDA_CALL(cudaMemcpy(dev_segmentChrom, SegmentChrom, SEGMENT_BP_NUM_META * sizeof(CoorBasePair), cudaMemcpyHostToDevice));
		//

		CoorBasePair *StraightChrom = (CoorBasePair*)malloc(sizeof(CoorBasePair)*STRAIGHT_BP_NUM_META);
		const char *straight = document["straightChromatinMeta"].GetString();
		printf("Straight Chromatin Table: Reading %s\n", straight);
		FILE *fpStraight = fopen(straight,"r");
		// float dump_float;
    	// int dump;
		// float bx, by, bz, rx, ry, rz, lx, ly, lz;
	    for (int i=0; i<STRAIGHT_BP_NUM_META; i++)
		{
		    fscanf(fpStraight,"%f %f %f %f %f %f %f %f %f %f\n", &dump_float, &bx, &by, &bz, &rx, &ry, &rz, &lx, &ly, &lz);
			dump = dump_float;
			//if(i<5) printf("%d %f %f %f %f %f %f %f %f %f\n", dump, bx, by, bz, rx, ry, rz, lx, ly, lz);
			StraightChrom[i].base.x = bx;
			StraightChrom[i].base.y = by;
			StraightChrom[i].base.z = bz;
			StraightChrom[i].right.x = rx;
			StraightChrom[i].right.y = ry;
			StraightChrom[i].right.z = rz;
			StraightChrom[i].left.x = lx;
			StraightChrom[i].left.y = ly;
			StraightChrom[i].left.z = lz;
		}
		fclose(fpStraight);
		CUDA_CALL(cudaMalloc((void**)&dev_straightChrom, STRAIGHT_BP_NUM_META * sizeof(CoorBasePair)));
		CUDA_CALL(cudaMemcpy(dev_straightChrom, StraightChrom, STRAIGHT_BP_NUM_META * sizeof(CoorBasePair), cudaMemcpyHostToDevice));

		CoorBasePair *BendChrom = (CoorBasePair*)malloc(sizeof(CoorBasePair)*BEND_BP_NUM_META);
		const char *bend = document["bentChromatinMeta"].GetString();
		printf("Bend Chromatin Table: Reading %s\n", bend);
	  FILE *fpBend = fopen(bend,"r");
	   for (int i=0; i<BEND_BP_NUM_META; i++)
		 {
		  fscanf(fpStraight,"%f %f %f %f %f %f %f %f %f %f\n", &dump_float, &bx, &by, &bz, &rx, &ry, &rz, &lx, &ly, &lz);
			dump = dump_float;
			//if(i<5) printf("%d %f %f %f %f %f %f %f %f %f\n", dump, bx, by, bz, rx, ry, rz, lx, ly, lz);
			BendChrom[i].base.x = bx;
			BendChrom[i].base.y = by;
			BendChrom[i].base.z = bz;
			BendChrom[i].right.x = rx;
			BendChrom[i].right.y = ry;
			BendChrom[i].right.z = rz;
			BendChrom[i].left.x = lx;
			BendChrom[i].left.y = ly;
			BendChrom[i].left.z = lz;
		 }
		fclose(fpBend);
		CUDA_CALL(cudaMalloc((void**)&dev_bendChrom, BEND_BP_NUM_META * sizeof(CoorBasePair)));
		CUDA_CALL(cudaMemcpy(dev_bendChrom, BendChrom, BEND_BP_NUM_META * sizeof(CoorBasePair), cudaMemcpyHostToDevice));
		
		float hisx, hisy, hisz;
		float3* bendHistone = (float3*)malloc(sizeof(float3)*BEND_HISTONE_NUM_META);
		const char *bent = document["bendHistone"].GetString();
		printf("Bent Histone Table: Reading %s\n", bent);
		FILE *fpBentH = fopen(bent,"r");
	    for (int i=0; i<BEND_HISTONE_NUM_META; i++)
		{
		    fscanf(fpBentH,"%f %f %f\n", &hisx, &hisy, &hisz);
		    //if(i<5) printf("%f %f %f\n", hisx, hisy, hisz);
			bendHistone[i].x = hisx;
			bendHistone[i].y = hisy;
			bendHistone[i].z = hisz;
		}
		fclose(fpBentH);
		CUDA_CALL(cudaMalloc((void**)&dev_bendHistone, BEND_HISTONE_NUM_META * sizeof(float3)));
		CUDA_CALL(cudaMemcpy(dev_bendHistone, bendHistone, BEND_HISTONE_NUM_META * sizeof(float3), cudaMemcpyHostToDevice));
		
		float3 *straightHistone = (float3*)malloc(sizeof(float3)*STRAIGHT_HISTONE_NUM_META);
		const char *straiHistone = document["straightHistoneMeta"].GetString();
		printf("Straight Histone Table: Reading %s\n", straiHistone);
		FILE *fpStraiH = fopen(straiHistone,"r");
	    for (int i=0; i<STRAIGHT_HISTONE_NUM_META; i++)
		{
		    fscanf(fpStraiH,"%f %f %f\n", &hisx, &hisy, &hisz);
		    //if(i<5) printf("%f %f %f\n", hisx, hisy, hisz);
			straightHistone[i].x = hisx;
			straightHistone[i].y = hisy;
			straightHistone[i].z = hisz;
		}
		fclose(fpStraiH);
		CUDA_CALL(cudaMalloc((void**)&dev_straightHistone, STRAIGHT_HISTONE_NUM_META * sizeof(float3)));
		CUDA_CALL(cudaMemcpy(dev_straightHistone, straightHistone, STRAIGHT_HISTONE_NUM_META * sizeof(float3), cudaMemcpyHostToDevice));
		
		free(StraightChrom);
		free(BendChrom);	
		free(bendHistone);	
		free(straightHistone);

		//modelTableSetup(dev_chromatinIndex,dev_chromatinStart,dev_chromatinType,dev_straightChrom,dev_bendChrom,dev_straightHistone,dev_bendHistone);
		printf("DNA geometry has been loaded to GPU memory\n");	 
 std::cout<<"finished initializing metaphase\n";
}





//chemsearch starts here

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
__device__ float3 pos2localMeta(int type, float3 pos, int index)
{
//do the coordinate transformation, index is the linear index for the referred box
//from global XYZ to local XYZ so that we can use the position of DNA base in two basic type (Straight and Bend) 
	int i = index%NUCLEUS_DIM_META;//the x,y,z index of the box
	int j = floorf((index%(NUCLEUS_DIM_META*NUCLEUS_DIM_META))/NUCLEUS_DIM_META);
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
	float shiftz = (k - (NUCLEUS_DIM_Z_META / 2)) * UNITLENGTH_META + UNITLENGTH_META * 0.5; 
	float shifty = (j - (NUCLEUS_DIM_META / 2)) * UNITLENGTH_META + UNITLENGTH_META * 0.5; 
	float shiftx = (i - (NUCLEUS_DIM_META / 2)) * UNITLENGTH_META + UNITLENGTH_META * 0.5; 
	pos.x = pos.x - shiftx; //relative to its center ?
	pos.y = pos.y - shifty; // 
	// pos.z = pos.z-(2*z + 1 - NUCLEUS_DIM_Z_META)*UNITLENGTH_META*0.5;
	pos.z = pos.z - shiftz;
	//printf("local coordinate %f %f %f\n", pos.x, pos.y, pos.z);
	// if (index == 27315) {
	// 	printf("It thinks the index is :: %d %d %d\n", i, j, k);
	// 	printf("So it shifts by %f %f %f\n", shiftx, shifty, shiftz);
	// }
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
	return (abs(a.x - b.x) <= CYLINDERRADIUS_META) 
		&& (abs(a.y - b.y) <= 50 + CYLINDERRADIUS_META * 2)
		&& (height_down <= a.z - b.z) 
		&& (a.z - b.z <= height_up);
}
__device__ bool withinCylinder(float3 &a, float3 &cylinder) {
	// check height
	// a = 7204.624023 -229.854507 14569.363281 :: 7171.703613 -365.000000 14413.875977
	if ((a.z < (cylinder.z - CYLINDERHEIGHT_META / 2)) || 
	    ((cylinder.z + CYLINDERHEIGHT_META / 2) < a.z)) {
		return 0;
	}
	// check radial distance
	if ((a.x - cylinder.x) * (a.x - cylinder.x) + 
		(a.y - cylinder.y) * (a.y - cylinder.y) > 
		CYLINDERRADIUS_META * CYLINDERRADIUS_META) {
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
		
		// if (id < 10) {
		// 	printf("Current e position :: %f %f %f\n",
		// 		pos_cur_target.x, pos_cur_target.y, pos_cur_target.z
		// 	);
		// }
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
		for (int i = NUMCHROMOSOMES_META - 1; i >= 0 ; i--) { // 46
			float height_up = (dev_chromosome_type[i] / 2) * CYLINDERHEIGHT_META + CYLINDERHEIGHT_META / 2;
			float height_down = -(((dev_chromosome_type[i] - 1) / 2) * CYLINDERHEIGHT_META + CYLINDERHEIGHT_META / 2);
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
		// if (id < 10) {
		// 	printf("ID chromosome: Rad ID = %d Chrom ID = %d %d\n", id, id_chromosome, found_nearest_chromosome);
		// }

		if (found_nearest_chromosome == 0) {
			// Step 1.2) if for this radical we did not find
			// anything nearby, then continue to the next :) 
			id+=blockDim.x*gridDim.x;
			continue ;
		}
		
		// if (id < 10) {
		// 	printf("ID chromosome %d %d\n", id, id_chromosome);
		// }
		// if we are here means we found chromosome
		// Step 1.3) Find nearest cylinder!
		int ttype = dev_chromosome_type[id_chromosome];
		int upper_part = ttype / 2;  // typy 4 :: 4 / 2 = 2 || type 5 :: 5 / 2 = 2
		int lower_part = (ttype - 1) / 2; // type 4 :: 4 / 2 - 1 = 1 || type 5 :: 5 / 2 - 1 = 1
		float3 nearest = dev_chromosome[id_chromosome];
		// if (id < 10) {
		// 	printf("middle :: Rad %f %f %f :: Nearest %f %f %f\n", pos_cur_target.x, pos_cur_target.y, pos_cur_target.z, 
		// 	nearest.x, nearest.y, nearest.z);
		// }
		int found_cylinder = 0;
		int id_cylinder = -1;
		// check lower and upper parts, cylinders
		// now we need to redo cylinder ID, to be in range [1 10] instead of [0 9]
		for (int idy = 0; idy < upper_part; idy++) {
			// Step 1.4) Check left and right cylinders 
			float3 left_shift;
			left_shift.x = nearest.x + 0.0;
			left_shift.y = nearest.y - 50 - CYLINDERRADIUS_META; 
			left_shift.z = nearest.z + CYLINDERHEIGHT_META * (idy + 1);
			float3 right_shift;
			right_shift.x = nearest.x + 0.0;
			right_shift.y = nearest.y + 50 + CYLINDERRADIUS_META; 
			right_shift.z = nearest.z + CYLINDERHEIGHT_META * (idy + 1);
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
				id_chromosome = id_chromosome + NUMCHROMOSOMES_META; // right side chromosome
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
				left_shift.y = nearest.y - 50 - CYLINDERRADIUS_META; 
				left_shift.z = nearest.z - CYLINDERHEIGHT_META * (idy + 1); // 
				float3 right_shift;
				right_shift.x = nearest.x + 0.0;
				right_shift.y = nearest.y + 50 + CYLINDERRADIUS_META; 
				right_shift.z = nearest.z - CYLINDERHEIGHT_META * (idy + 1);
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
					id_chromosome = id_chromosome + NUMCHROMOSOMES_META; // right side chromosome
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
			left_shift.y = nearest.y - CYLINDERRADIUS_META; 
			left_shift.z = nearest.z + 0.0;
			float3 right_shift;
			right_shift.x = nearest.x + 0.0;
			right_shift.y = nearest.y + CYLINDERRADIUS_META; 
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
				id_chromosome = id_chromosome + NUMCHROMOSOMES_META; // right side chromosome
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
		
		// printf("The shifted e position :: %f %f %f\n", 
		// 	pos_cur_target.x, pos_cur_target.y, pos_cur_target.z
		// );
		// printf("Chromosome ID :: %d and Cylinder ID :: %d\n", 
		// 	id_chromosome, id_cylinder
		// );
		// from the global coordinate (-min max) to [0 N] index coordinate
		// what we know is that z must be say 6
		index.x=floorf(pos_cur_target.x/UNITLENGTH_META) + (NUCLEUS_DIM_META/2); // 2000 
		index.y=floorf(pos_cur_target.y/UNITLENGTH_META) + (NUCLEUS_DIM_META/2);
		index.z=floorf(pos_cur_target.z/UNITLENGTH_META) + (NUCLEUS_DIM_Z_META/2);
		
		// printf("It thinks Nucleosome index is %d %d %d\n", 
		// 	index.x, index.y, index.z
		// );

		int delta=index.x+index.y*NUCLEUS_DIM_META+index.z*NUCLEUS_DIM_META*NUCLEUS_DIM_META,minindex=-1;
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
			if(newindex<0 || newindex > NUCLEUS_DIM_META*NUCLEUS_DIM_META*NUCLEUS_DIM_Z_META-1) continue;
			int type = dev_chromatinType[newindex];
			// if (i == 13) { 
			// 	printf("Type check %d\n", type);
			// }
			if(type==-1 || type==0) continue;

			newpos = pos2localMeta(type, pos_cur_target, newindex);
			if(type<7)
			{
				// if(newpos.x<(min1-SPACETOBODER) || newpos.y<(min2-SPACETOBODER) || newpos.z<(min3-SPACETOBODER) ||newpos.x>(max1+SPACETOBODER)
				//  || newpos.y>(max2+SPACETOBODER) || newpos.z>(max3+SPACETOBODER))
				// 	continue;
				chrom=dev_straightChrom;
				chromNum=STRAIGHT_BP_NUM;
				histone=dev_straightHistone;
				histoneNum=STRAIGHT_HISTONE_NUM;
			}
			else
			{
				// if(newpos.x<(min1-SPACETOBODER) || newpos.y<(min2-SPACETOBODER) || newpos.z<(min3-SPACETOBODER) ||newpos.x>(max3+SPACETOBODER)
				//  || newpos.y>(max2+SPACETOBODER) || newpos.z>(max1+SPACETOBODER))
				// 	continue;
				chrom=dev_bendChrom;
				chromNum=BEND_BP_NUM;
				histone=dev_bendHistone;
				histoneNum=BEND_HISTONE_NUM;
			}
			// for(int j=0;j<histoneNum;j++)
			// {
			// 	mindis = caldistance(newpos, histone[j])-RHISTONE;
			// 	if(mindis < 0) flag=1;
			// }
			if(flag) break;
			for(int j=0;j<chromNum;j++) // 200 nucleosome
			{
				// can take the size of base into consideration, distance should be distance-r;
				mindis=100,minindex=-1;
				distance[0] = caldistance(newpos, chrom[j].base)-RBASE;
				distance[1] = caldistance(newpos,chrom[j].left)-RSUGAR;
				distance[2] = caldistance(newpos,chrom[j].right)-RSUGAR;
				// if (i == 13 && j == 99) { // lol 100th 
				// 	printf("Event within voxel %f %f %f \n and Right base pair :: %f %f %f \n",
				// 		newpos.x, newpos.y, newpos.z,
				// 		chrom[j].right.x, chrom[j].right.y, chrom[j].right.z
				// 	);
				// }
				for(int iii=0;iii<3;iii++)
				{
					if(mindis>distance[iii])
					{
						mindis=distance[iii];
						minindex=iii;
					}
				}
				// if (i == 13 && j == 99) { // lol 100th 
				// 	printf("13 and 99 :: Distances are %f %f %f\n", distance[0], distance[1], distance[2]);
				// 	printf("Min of these and id of this :: %f %d\n", mindis, minindex);
				// }
				if(mindis<0)
				{
					if(minindex>0)
					{
						// if (i == 13 && j == 99) { // lol 100th 
						// 	printf("Technically Recorded 13, 99\n");
						// }
						// printf("Ids: starting bp, inside bp, cyl = %d %d %d\n", dev_chromatinStart[newindex], j, id_cylinder);
						// GEANT4  
						d_recorde[id].site.x = id_chromosome; // 
						d_recorde[id].site.y = (dev_chromatinStart[newindex]+j) + TOTALBP_META * id_cylinder;  
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
						d_recorde[id].site.y = (dev_chromatinStart[newindex]+j) + TOTALBP_META * id_cylinder;  
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
						newdelta = delta + NUCLEUS_DIM_META*NUCLEUS_DIM_META;
					else if (i < 20) // xz +1y
						newdelta = delta + NUCLEUS_DIM_META;
					else // yz +1x
						newdelta = delta + 1;
				}
				int newindex = newdelta * 12 + i % 12;

				// if (i == 13) {
				// 	printf("ID check %d %d\n", newindex, delta);
				// }

				// ************flag changed Z
				// printf("New index vs total volume :: %d vs %d\n", newindex, NUCLEUS_DIM*NUCLEUS_DIM*NUCLEUS_DIM_Z * 12);
				if(newindex<0 || newindex >= NUCLEUS_DIM_META*NUCLEUS_DIM_META*NUCLEUS_DIM_Z_META * 12) continue;
				
				int type = dev_segmentType[newindex];
				// if (i == 13) { 
				// 	printf("Type check %d\n", type);
				// }
				// printf("Type = %d\n", type);
				if(type==-1 || type==0) continue;

				// type is not used for pos2local because we are just getting next to this cell
				// no rotation is needed here
				float3 pos_within_voxel = pos2localMeta(1, pos_cur_target, newdelta);
				// the idea here is to shift first the position within the voxel?
				// shift relative to the voxel
				// then choose to shift next to wall center?
				// 
				newpos = PosToWall(type, pos_within_voxel, i % 12);
				// printf("Id %d and relative poistion: %0.2f %0.2f %0.2f\n", i, newpos.x, newpos.y, newpos.z);
				if(type<7)
				{
					// if(newpos.x<(min1-SPACETOBODER) || newpos.y<(min2-SPACETOBODER) || newpos.z<(min3-SPACETOBODER) ||newpos.x>(max1+SPACETOBODER)
					// || newpos.y>(max2+SPACETOBODER) || newpos.z>(max3+SPACETOBODER))
					// 	continue;
					chrom=dev_segmentChrom;
					chromNum=SEGMENT_BP_NUM_META;
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
					// if (i == 13 && j == 99) { // lol 100th 
					// 	printf("13 and 99 :: Distances are %f %f %f\n", distance[0], distance[1], distance[2]);
					// 	printf("Min of these and id of this :: %f %d\n", mindis, minindex);
					// }
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
							d_recorde[id].site.y = (dev_segmentStart[newindex]+j) + TOTALBP_META * id_cylinder;  
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
							d_recorde[id].site.y = (dev_segmentStart[newindex]+j) + TOTALBP_META * id_cylinder;  
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
