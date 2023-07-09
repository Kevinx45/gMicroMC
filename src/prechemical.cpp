#include "prechemical.h"

PrechemList::PrechemList()
{
	readBranchInfo(document["fileForBranchInfo"].GetString());
	readThermRecombInfo(document["fileForRecombineInfo"].GetString());
	readWaterStates();
}

PrechemList::~PrechemList()
{

	free(electron_container);
	free(ion_container);
	free(ion_tag);
	free(ion_index);
	free(elec_index);
}

void PrechemList::readBranchInfo(std::string fname)
{
	char buffer[256];
	FILE *fp = fopen(fname.c_str(), "r");   
    printf("\n\nloading %s\n", fname.c_str());
	
	fgets(buffer, 250, fp);	
	fscanf(fp, "%d\n", &nbrantype);
	fgets(buffer, 250, fp);	
	fscanf(fp, "%d\n", &max_prod_bran);
	cudaMallocManaged(&num_prod_bran,sizeof(int) * nbrantype);
	cudaMallocManaged(&prodtype_bran,sizeof(int) * nbrantype * max_prod_bran);	

	fgets(buffer, 250, fp);
	int temp, i, k;	
	for(i=0; i<nbrantype; i++)
	{
	  fscanf(fp, "%d %d", &temp, &num_prod_bran[i]);
	  for(k=0; k<num_prod_bran[i]; k++)
	  {
	    fscanf(fp, "%d", &prodtype_bran[i* max_prod_bran + k]);
	  }
	}
	fscanf(fp, "\n");

	fgets(buffer, 250, fp);
	fscanf(fp, "%d\n", &num_replace_bran);

	cudaMallocManaged(&para_replace_bran,sizeof(float) * nbrantype * num_replace_bran);
	fgets(buffer, 250, fp);
    for(i=0; i<nbrantype; i++)
	{
	    fscanf(fp, "%d", &temp);
		for(k=0; k<num_replace_bran; k++)
		{
		    fscanf(fp, "%f", &para_replace_bran[i*num_replace_bran+k]);
		}
		fscanf(fp, "\n");
	}	
	

	fgets(buffer, 250, fp);	
	fscanf(fp, "%d\n", &nparentype);
	fgets(buffer, 250, fp);	
	fscanf(fp, "%d\n", &max_bran_paren);

	cudaMallocManaged(&num_bran_paren,sizeof(int) * nparentype);
	cudaMallocManaged(&brantype_paren,sizeof(int) * nparentype * max_bran_paren);
	cudaMallocManaged(&branratio_paren,sizeof(float) * nparentype * max_bran_paren);
	
	for(i=0; i<nparentype; i++)
	{
	  fgets(buffer, 250, fp);
	  fscanf(fp, "%d %d", &temp, &num_bran_paren[i]);
	  
	  for(k=0; k<num_bran_paren[i]; k++)
	  {
	    fscanf(fp, "%d %f", &brantype_paren[i* max_bran_paren + k], &branratio_paren[i* max_bran_paren + k]);
	   }
	  fscanf(fp, "\n");
	}


	if(verbose>1)
	{
		printf("Information is listed in the following\n");
		printf("number of branches\n%d\n",nbrantype);
		for(i =0; i<nbrantype;i++)
		{
			printf("type %d has %d products: ", i, num_prod_bran[i]);
			for(k=0;k<num_prod_bran[i];k++)
			{
				printf("%d ", prodtype_bran[i*max_prod_bran+k]);
			}
			printf("\n");
			printf("type %d has %d parameters for product displacement: ", i, num_replace_bran);
			for(k=0;k<num_replace_bran;k++)
			{
				printf("%f ", para_replace_bran[i*num_replace_bran+k]);
			}
			printf("\n");
		}
		
		printf("number of parent molecule types: %d\n", nparentype);
		printf("maximum branch types per parent molecule: %d\n", max_bran_paren);
		for(i =0; i<nparentype;i++)
		{
			printf("parent type %d has %d number of branches: ", i, num_bran_paren[i]);
			for(k=0;k<num_bran_paren[i];k++)
			{
				printf("%d %f ", brantype_paren[i*max_bran_paren+k], branratio_paren[i* max_bran_paren + k]);
			}
			printf("\n");
		}
	}
}

void PrechemList::readThermRecombInfo(std::string fname)
{
	char buffer[256];
	FILE *fp = fopen(fname.c_str(), "r");   
    printf("\n\nloading %s\n", fname.c_str());
	
	fgets(buffer, 250, fp);
	fscanf(fp, "%f\n", &Ecut_recom);	
	
	fgets(buffer, 250, fp);
	fscanf(fp, "%d %d\n", &num_para_recom[0],&num_para_recom[1]);

	cudaMallocManaged(&pro_recom, sizeof(float) * num_para_recom[0]);	
	cudaMallocManaged(&rms_therm_elec, sizeof(float) * num_para_recom[1]);

	fgets(buffer, 250, fp);
	for(int i=0; i<num_para_recom[0]; i++)
	{
	    fscanf(fp, "%f", &pro_recom[i]);
	}
	fscanf(fp, "\n");
	fgets(buffer, 250, fp);
	for(int i=0; i<num_para_recom[1]; i++)
	{
	    fscanf(fp, "%f", &rms_therm_elec[i]);
	}
	fclose(fp);

	if(verbose>1)
	{
		printf("Information is listed in the following\n");
		printf("There are %d parameters for the electron hole recombination probability\n",num_para_recom[0]);
		printf("probability parameters: ");
		for(int i=0;i<num_para_recom[0];i++)
		{
			printf("%f ",pro_recom[i]);
		}
		printf("\n");
		printf("There are %d parameters for the electron migration rms calculation\n",num_para_recom[0]);
		printf("rms parameters: ");
		for(int i=0;i<num_para_recom[1];i++)
		{
			printf("%f ",rms_therm_elec[i]);
		}
		printf("\n");
	}
}

void PrechemList::readWaterStates()
{
	std::string fname = document["fileForIntInput"].GetString();
	FILE* fpint=fopen(fname.c_str(), "rb");
    fname = document["fileForFloatInput"].GetString();
	FILE* fpfloat=fopen(fname.c_str(), "rb");
	int start, stop;
	start = ftell(fpfloat);
	fseek (fpfloat, 0, SEEK_END);
	stop = ftell(fpfloat);
	fseek (fpfloat, 0, SEEK_SET);
	printf("water state start=%d, end=%d\n",start,stop);
	num_total_paren = (stop-start)/4/5;

	int* phyint = (int*)malloc(sizeof(int) * num_total_paren*4);
	float* phyfloat = (float*)malloc(sizeof(float) * num_total_paren*5);
	if(phyfloat && phyint)
	{
		fread(phyint,sizeof(int),4*num_total_paren,fpint);
		fread(phyfloat,sizeof(float),5*num_total_paren,fpfloat);
	}
	else
	{
		printf("Wrong input!!!\n");
		exit(1);
	}
	int num_elec, num_wi,num_we_a1b1, num_we_b1a1,num_we_rd, num_w_dis; //number of solvated electrons, etc.

	fread(&num_elec,sizeof(int),1,fpint);
	fread(&num_wi,sizeof(int),1,fpint);
	fread(&num_we_a1b1,sizeof(int),1,fpint);
	fread(&num_we_b1a1,sizeof(int),1,fpint);
	fread(&num_we_rd,sizeof(int),1,fpint);
	fread(&num_w_dis,sizeof(int),1,fpint);

	fclose(fpint);
	fclose(fpfloat);
	
	printf("the total number of initial reactant is %d\n", num_total_paren);
	printf("num_elec=%d, num_wi=%d, num_we_a1b1=%d, num_we_b1a1=%d, num_we_rd=%d, num_w_dis=%d\n", num_elec, num_wi, num_we_a1b1, num_we_b1a1, num_we_rd, num_w_dis);
	
	cudaMallocManaged(&type_paren, sizeof(int) * num_total_paren*3);
	cudaMallocManaged(&posx_paren, sizeof(float) * num_total_paren*3);
	cudaMallocManaged(&posy_paren, sizeof(float) * num_total_paren*3);
	cudaMallocManaged(&posz_paren, sizeof(float) * num_total_paren*3);
	cudaMallocManaged(&ene_paren, sizeof(float) * num_total_paren);
	cudaMallocManaged(&ttime_paren, sizeof(float) * num_total_paren*3);
	cudaMallocManaged(&index_paren, sizeof(int) * num_total_paren*3);

	for (int l=0;l<num_total_paren*3;l++) //for (int l=num_total_paren;l<num_total_paren*3;l++) Change 20-03-2023
	{
		type_paren[l]=255;
	}

	int eleN=5;
	electron_container=(float*)malloc(sizeof(float)*eleN*(num_elec+num_w_dis));
 	ion_container=(float*)malloc(sizeof(float)*eleN*num_wi);	
 	ion_tag=(int*)malloc(sizeof(int)*num_wi);
 	ion_index=(int*)malloc(sizeof(int)*num_wi);
 	elec_index=(int*)malloc(sizeof(int)*(num_elec+num_w_dis));
	
	int idx_elec = 0;
	int idx_wi = 0;
	int idx_excite=0;
	int ptype,stype,index, tempnum;
	float tempe,tempx,tempy,tempz,tempt;
	float temprand;
	for(int i = 0; i < num_total_paren; i++)
	{
        if (i%10000==0)printf("i = %d, %d\n", i, num_total_paren);

		index=phyint[4*i+1];
		ptype=phyint[4*i+2];
		stype=phyint[4*i+3];
		tempe=phyfloat[5*i];
		tempx=phyfloat[5*i+1];
		tempy=phyfloat[5*i+2];
		tempz=phyfloat[5*i+3];
		tempt = phyfloat[5*i+4];
        
		if(ptype == 7) //water molecule
		{
		  if(stype <= 4) // ionized water molecule
		  {
		    ion_container[idx_wi*eleN]=tempe;
		    ion_container[idx_wi*eleN+1] = tempx;
			ion_container[idx_wi*eleN+2] = tempy;
			ion_container[idx_wi*eleN+3] = tempz;
			ion_container[idx_wi*eleN+4] = tempt;
			ion_tag[idx_wi]=1;
			ion_index[idx_wi]=index;
			idx_wi++;
		  }
		  else if (stype<10)
		  {
		    posx_paren[idx_excite] = tempx;
			posy_paren[idx_excite] = tempy;
			posz_paren[idx_excite] = tempz;
			ene_paren[idx_excite] = tempe;
			ttime_paren[idx_excite] = tempt;
			index_paren[idx_excite]=index;
			if(stype == 5) //a1b1
		  	{
				type_paren[idx_excite] = 0;

		  	}
		   	else if(stype == 6) //b1a1
		  	{
				type_paren[idx_excite] = 1;
		  	}
		   	else if(stype >= 7 && stype <= 9) //rydberg and diffusion bands
		  	{
				type_paren[idx_excite] = 2;
		  	}
		  	idx_excite++;
		  }
		 else if(stype == 10) 
		  {
				temprand=(float)rand()/INT_MAX;
				//printf("temprand=%f\n", temprand);
				if (temprand<0.1)// dissociative electron attachment
				{
		    		posx_paren[idx_excite] = tempx;
					posy_paren[idx_excite] = tempy;
					posz_paren[idx_excite] = tempz;
					ene_paren[idx_excite] = tempe;
					ttime_paren[idx_excite] = tempt;
					index_paren[idx_excite]=index;
					type_paren[idx_excite] = 3;
					idx_excite++;
				}
				else
				{
					electron_container[idx_elec*eleN]=tempe;
		    		electron_container[idx_elec*eleN+1] = tempx;
					electron_container[idx_elec*eleN+2] = tempy;
					electron_container[idx_elec*eleN+3] = tempz;
					electron_container[idx_elec*eleN+4] = tempt;
					elec_index[idx_elec]=index;
					idx_elec++;
					//printf("idx_elec=%d, eleN=%d\n", idx_elec, eleN);
				}
		  }	  
		  	
		} 
		else if(ptype == 0) //solvated electron
		{		  
		    electron_container[idx_elec*eleN]=tempe;
		    electron_container[idx_elec*eleN+1] = tempx;
			electron_container[idx_elec*eleN+2] = tempy;
			electron_container[idx_elec*eleN+3] = tempz;
			electron_container[idx_elec*eleN+4] = tempt;
			elec_index[idx_elec]=index;
			idx_elec++;
		}
	}

	if(idx_wi != num_wi || idx_excite +idx_elec!=num_we_a1b1+num_we_b1a1+num_we_rd+num_w_dis+num_elec)
	{
	    printf("error in the number of the initial particles for prechemical stage.\n");
		exit(1);
	}

	float minidis;
	float pro;
	int idx_recom=0;
	int idx_mini=0;
	float tx,ty,tz,tempdis;
	for (int i = 0; i < idx_elec; i++)
	{
		tempe=electron_container[i*eleN];
		tempx=electron_container[i*eleN+1];
		tempy=electron_container[i*eleN+2];
		tempz=electron_container[i*eleN+3];
		tempt=electron_container[i*eleN+4];
		posx_paren[idx_excite+i] = tempx;
		posy_paren[idx_excite+i] = tempy;
		posz_paren[idx_excite+i] = tempz;
		ene_paren[idx_excite+i] = tempe;
		ttime_paren[idx_excite+i] = tempt;
		type_paren[idx_excite+i] = 5; // hydrated electron
		index_paren[idx_excite+i]=elec_index[i];
		if (tempe<Ecut_recom)
		{
			pro=0;

			for (int j=0;j<num_para_recom[1];j++)
			{
				pro+=pro_recom[j]*pow(tempe,num_para_recom[1]-j-1);
			}
			temprand=(float)rand()/INT_MAX;
			
			if (temprand<pro)
			{
				minidis=100000.0f;
				for (int j=0;j<idx_wi;j++)
				{
					tx=ion_container[j*eleN+1];
					ty=ion_container[j*eleN+2];
					tz=ion_container[j*eleN+3];
					tempdis=sqrt(pow(tempx-tx,2)+pow(tempy-ty,2)+pow(tempz-tz,2));
					if (tempdis<minidis && ion_tag[j]==1)
					{ 
						minidis=tempdis;
						idx_mini=j; 
					}
				}
				posx_paren[idx_excite+i] = ion_container[idx_mini*eleN+1];
				posy_paren[idx_excite+i] = ion_container[idx_mini*eleN+2];
				posz_paren[idx_excite+i] = ion_container[idx_mini*eleN+3];
				ttime_paren[idx_excite+i] = ion_container[idx_mini*eleN+4];
				type_paren[idx_excite+i] = 4; // recombined electron-hole
				ion_tag[idx_mini]=255;
				idx_recom++;
				//printf("idx_mini=%d, idx_recom=%d\n",idx_mini, idx_recom);
				
			}
		}
	}
	printf("idx_elec = %d, idx_wi = %d, idx_excite = %d, idx_recom = %d, num_wi=%d\n", idx_elec-idx_recom, idx_wi, idx_excite, idx_recom, num_wi);
	idx_wi=0;
	for (int i = 0; i < num_wi; i++)
	{
		if (ion_tag[i]==1)
		{
			tempe=ion_container[i*eleN];
			tempx=ion_container[i*eleN+1];
			tempy=ion_container[i*eleN+2];
			tempz=ion_container[i*eleN+3];
			tempt=ion_container[i*eleN+4];
			posx_paren[idx_excite+idx_elec+idx_wi] = tempx;
			posy_paren[idx_excite+idx_elec+idx_wi] = tempy;
			posz_paren[idx_excite+idx_elec+idx_wi] = tempz;
			ene_paren[idx_excite+idx_elec+idx_wi] = tempe;
			ttime_paren[idx_excite+idx_elec+idx_wi] = tempt;
			type_paren[idx_excite+idx_elec+idx_wi] = 6; // ionized water molecule
			index_paren[idx_excite+idx_elec+idx_wi]=ion_index[i];
			idx_wi++;
		}

	}
			
	printf("idx_elec = %d, idx_wi = %d, idx_excite = %d, idx_recom = %d\n", idx_elec-idx_recom, idx_wi, idx_excite, idx_recom);
	// need to substract the recombined electron from the total count
	num_total_paren -=idx_recom;
	if(idx_wi+idx_recom!= num_wi)
	{
	    printf("idx_wi=%d, idx_recom=%d, num_wi=%d\n",idx_wi,idx_recom,num_wi);
	    printf("error in the number of the recombined hole and residue hole for prechemical stage.\n");
		exit(1);
	}
	for (int i=0;i<10;i++)
	{
		printf("particle %d has posx=%f, posy=%f, posz=%f, type_paren=%d, ene_paren=%f \n", i, posx_paren[i], posy_paren[i], posz_paren[i], type_paren[i], ene_paren[i]);
	}
		int num_electest=0;
	for (int i=0;i<num_total_paren;i++)
	{
		if (type_paren[i]==5)
		{num_electest++;};
	}
	printf("num_electest=%d\n", num_electest);
	free(phyint);
	free(phyfloat);
}


