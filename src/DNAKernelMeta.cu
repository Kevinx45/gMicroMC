#include "DNAKernelMeta.cuh"
__constant__  int neighborindex[27];
__constant__ float min1, min2, min3, max1, max2, max3;
__constant__  float d_rDNA[72];
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
    
    int* dev_chromatinIndex;
		int* dev_chromatinStart;
		int* dev_chromatinType;
		CoorBasePair* dev_straightChrom;
		CoorBasePair* dev_segmentChrom;
		CoorBasePair* dev_bendChrom;
		float3* dev_straightHistone;
		float3* dev_bendHistone;
		float3* dev_chromosome;
    int *dev_chromosome_type;
		// allocating space for the segments connecting nucleosomes
		int* dev_segmentIndex;
		int* dev_segmentStart;
		int* dev_segmentType;


    // X-CHROMOSOMES, there are 46 of them 
    int data[6];
		//cout << "Reading the chromosomes and types?\n";
		ifstream fin;
		fin.open("../tables/metadna/chromosome_coordinates_v6.txt"); // v5 has 0,0,0 chromosome
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
		
		cout << "Time to read voxelized coordinates \n";
		//ifstream fin;
		// I need to figure out how to get extra coordinates
		// maybe I can store in the chromatin index as I'm not using it anyway
		fin.open(SIDESFILE);
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
				array <int, 2> subs[4] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
				// ids [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
				int subvoxel_id = 0;
				xxx -= 2; // center subvoxel
				yyy -= 2;
				zzz -= 2;
				array <int, 2> nn;
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
		cout << "end of reading voxelized coordinates \n\n";
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
		const char *segment = "./table/NucleosomeTableSegment.txt";
		printf("Straight Chromatin Table: Reading %s\n", segment);
		FILE *fpSegment = fopen(segment,"r");
		float dump_float;
    	int dump;
		float bx, by, bz, rx, ry, rz, lx, ly, lz;
	    for (int i=0; i<SEGMENT_BP_NUM_META; i++)
		{
		    fscanf(fpSegment,"%f %f %f %f %f %f %f %f %f %f\n", &dump_float, &bx, &by, &bz, &rx, &ry, &rz, &lx, &ly, &lz);
			dump = dump_float;
			//if(i<5) printf("%d %f %f %f %f %f %f %f %f %f\n", dump, bx, by, bz, rx, ry, rz, lx, ly, lz);
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
		const char *straight = "./table/NucleosomeTable200StraightZ.txt";
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
		const char *bend = "./table/NucleosomeTable200SideZ.txt";
		printf("Bend Chromatin Table: Reading %s\n", bend);
		FILE *fpBend = fopen(bend,"r");
	    for (int i=0; i<BEND_BP_NUM; i++)
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
		const char *bent = "./table/BentHistonesTable1.txt";
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
		const char *straiHistone = "./table/StraightHistonesTable1.txt";
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
cout<<"finished initializing metaphase"<<endl;
}
