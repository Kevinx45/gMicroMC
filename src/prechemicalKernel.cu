#include "prechemicalKernel.cuh"
#include "prechemical.h"

float *d_posx, *d_posy, *d_posz; // the GPU variables to store the positions of the particles (a larger memory is required to include the product of prechemical stage) 
float *d_ene_paren, *d_ttime; // initial energies of the initial particles
int *d_ptype, *d_index; // the species type of the particles (255 for empty entries or produced H2O)	
int *d_num_prod_bran, *d_prodtype_bran;
float *d_para_replace_bran;
int *d_num_bran_paren, *d_brantype_paren;
float *d_branratio_paren;
float *d_rms_therm_elec;

__device__ __constant__	int d_num_total; 
__device__ __constant__	int d_nbrantype;
__device__ __constant__	int d_max_prod_bran; 
__device__ __constant__	int d_num_replace_bran;


__device__ __constant__	int d_nparentype;
__device__ __constant__	int d_max_bran_paren; 


__device__ __constant__	int d_num_rms_para;
	

__global__ void physiochemical_decay(float *d_posx, // x position of the particles (input and output)
                                    float *d_posy,
									float *d_posz,
									int *d_ptype, // species type for products of prechemical stage, 255 for empty or produced water (output)
									int *d_num_bran_paren,
									float *d_ratio_bran_paren,
									int *d_brantype_paren,
									int *d_num_prod_bran,
									float *d_ene_paren,
									float *d_rms_therm_elec,
									float *d_para_replace_bran,
									int *d_prodtype_bran
									)
{											 
    const int tid = blockIdx.x*blockDim.x+ threadIdx.x;
	const int pid = tid;
	
	if(tid < d_num_total)
	{  
	    curandState localState = cuseed[pid];
		float radnum= curand_uniform(&localState);
		float pro=0.0f;
	    int parentype = d_ptype[tid];
	    if (tid<10)
			{
				printf("test 2: thread id=%d, d_num_total=%d, d_posx=%f, parentype=%d\n",tid, d_num_total,d_posx[tid],parentype);
			}
	    int numbran=d_num_bran_paren[parentype];
	    int brantype;
		int numprod;

		for (int i=0;i<numbran;i++)
		{
			pro+=d_ratio_bran_paren[d_max_bran_paren*parentype+i];

			if (radnum<pro)
			{			
	    		brantype=d_brantype_paren[d_max_bran_paren*parentype+i]; // branch type for that parent molecule
	    		numprod=d_num_prod_bran[brantype]; // number of product for that branch
	    		if (numprod==0)
	    		{
	    			d_ptype[tid]=255; // H20
	    		}
	    		else
	    		{
	    			// below to sample the displacement of the products
	    			float rms[2];
	    			float r[2];
	    			for (int j=0;j<2;j++)
	    			{
	    				rms[j]=0;
	    				r[j]=0;
	    			}
	    			float dir[6];
	    			for (int j=0;j<6;j++) dir[j]=0;
	    			float ene=d_ene_paren[tid];
	    			int flag=0;
	    			if (brantype==6) // parent type 5, branch type 6, hydrated electron, note: this is input file dependent (branchInfo_prechem.txt)
	    			{
	    				float rms0=0.0;
	    				for (int j=0;j<d_num_rms_para;j++)
	    				{
	    					rms0+=d_rms_therm_elec[j]*pow(ene,d_num_rms_para-j-1);
	    				}
	    				rms[0]=rms0;
	    				flag=1;	    			
	    			}
	    			else 
	    			{
	    				for (int j=0;j<2;j++)
	    				{
	    					rms[j]=d_para_replace_bran[brantype*d_num_replace_bran+j];
	    				}
	    				
	    			}
	    			float tempr, nx, ny, nz;
	    			for (int j=0;j<2;j++)
	    			{
	    				if (rms[j]!=0)
	    				{		
	    					get_distance(&localState, rms[j], &tempr,ene,flag);
	    					r[j]=tempr;
	    					get_direction(&localState, &nx, &ny, &nz);
	    					dir[3*j]=nx;
	    					dir[3*j+1]=ny;
	    					dir[3*j+2]=nz;
	    				}	    				
	    			}

					float tempx=d_posx[tid];
					float tempy=d_posy[tid];
					float tempz=d_posz[tid];
	    			float randnum = curand_uniform(&localState);
	    			if ((brantype==1 || brantype==3) &&randnum<0.5) // switch the two product positions
	    			{
	    				d_posx[tid]=tempx+d_para_replace_bran[d_num_replace_bran*brantype+4]*r[0]*dir[0]+d_para_replace_bran[d_num_replace_bran*brantype+5]*r[1]*dir[3];
	    				d_posy[tid]=tempy+d_para_replace_bran[d_num_replace_bran*brantype+4]*r[0]*dir[1]+d_para_replace_bran[d_num_replace_bran*brantype+5]*r[1]*dir[4];
	    				d_posz[tid]=tempz+d_para_replace_bran[d_num_replace_bran*brantype+4]*r[0]*dir[2]+d_para_replace_bran[d_num_replace_bran*brantype+5]*r[1]*dir[5];
	    				d_posx[tid+d_num_total]=tempx+d_para_replace_bran[d_num_replace_bran*brantype+2]*r[0]*dir[0]+d_para_replace_bran[d_num_replace_bran*brantype+3]*r[1]*dir[3];
	    				d_posy[tid+d_num_total]=tempy+d_para_replace_bran[d_num_replace_bran*brantype+2]*r[0]*dir[1]+d_para_replace_bran[d_num_replace_bran*brantype+3]*r[1]*dir[4];
	   					d_posz[tid+d_num_total]=tempz+d_para_replace_bran[d_num_replace_bran*brantype+2]*r[0]*dir[2]+d_para_replace_bran[d_num_replace_bran*brantype+3]*r[1]*dir[5]; 
	   					if (brantype==3)
	   					{
	   						get_electron_distance(&localState, &tempr);
    						get_direction(&localState, &nx, &ny, &nz);	    						
    						d_posx[tid+d_num_total*2]=tempx+tempr*nx;
	    					d_posy[tid+d_num_total*2]=tempy+tempr*ny;
	    					d_posz[tid+d_num_total*2]=tempz+tempr*nz; 
	    				}
	    				for (int j=0;j<numprod;j++) d_ptype[tid+d_num_total*j]=d_prodtype_bran[d_max_prod_bran*brantype+j]; // product type
	    				if (tid<100)
						{
							printf("test 5: thread id=%d, brantype=%d, parentype=%d, randnum=%f, r1=%f, r2=%f, tempr=%f, x1=%f, x2=%f, x3=%f\n",tid,brantype,parentype, randnum, r[0],d_para_replace_bran[d_num_replace_bran*brantype+4],d_para_replace_bran[d_num_replace_bran*brantype+5],d_posx[tid],d_posx[tid+d_num_total],d_posx[tid+d_num_total*2]);
						}
	    			}	    			
	    			else
	    			{
	    				for (int j=0;j<numprod;j++)
	    				{
	    					d_ptype[tid+d_num_total*j]=d_prodtype_bran[d_max_prod_bran*brantype+j]; // product type
		    				d_posx[tid+d_num_total*j]=tempx+d_para_replace_bran[d_num_replace_bran*brantype+2*(j+1)]*r[0]*dir[0]+d_para_replace_bran[d_num_replace_bran*brantype+2*(j+1)+1]*r[1]*dir[3];
	    					d_posy[tid+d_num_total*j]=tempy+d_para_replace_bran[d_num_replace_bran*brantype+2*(j+1)]*r[0]*dir[1]+d_para_replace_bran[d_num_replace_bran*brantype+2*(j+1)+1]*r[1]*dir[4];
	    					d_posz[tid+d_num_total*j]=tempz+d_para_replace_bran[d_num_replace_bran*brantype+2*(j+1)]*r[0]*dir[2]+d_para_replace_bran[d_num_replace_bran*brantype+2*(j+1)+1]*r[1]*dir[5];    					
	    				}
	    			}
	    		}
	    		break;
			}
		}  

        cuseed[pid] = localState;		
    }
}

__device__ void get_distance(curandState *localState_pt, float rms, float *r,float ene,int flag)
{
    float sigma=rms/sqrt(3.0); //https://doi.org/10.1016/j.ejmp.2015.10.087
    float pro_max;
    float r_max;
    if (flag==1 && ene<2) // electron < 2eV following f(r)=r^2/(2*sigma^3)*exp(-r/sigma)
    {
    	pro_max=2.0/sigma*exp(-2.0); // can be computed from the f(r)'=0
    	r_max=8.0*sigma;
    }
	else
	{
		pro_max=sqrt(2.0/PI)*2.0/sigma*exp(-1.0); //f(r)=sqrt(2/pi)*r^2/(sigma^3)*exp(-r^2/(2*sigma^2))
		r_max=3.75*sigma;
	}
	float rsample=curand_uniform(localState_pt)*r_max;
	float pro=1.0;

	float pror=0.0;

	while(pro>pror)
	{
		rsample=curand_uniform(localState_pt)*r_max;
		if (flag==1 && ene<2) // electron < 2eV following f(r)=r^2/(2*sigma^3)*exp(-r/sigma)
    	{
    		pror=pow(rsample,2)/(2.0*pow(sigma,3))*exp(-rsample/sigma);
    	}
		else
		{
			pror=sqrt(2.0/PI)*pow(rsample,2)/pow(sigma,3)*exp(-rsample*rsample/(2*sigma*sigma));
		}
		pro=curand_uniform(localState_pt)*pro_max;
	}
	*r=rsample;
}
__device__ void get_electron_distance(curandState *localState_pt, float *r)
{
    float r_max=0.5f; //https://doi.org/10.1016/j.ejmp.2015.10.087 f(r)=4*r*exp(-2r)
    float pro_max=2.0*exp(-1.0);
	float rsample=curand_uniform(localState_pt)*r_max;
	float pro=1.0;
	float pror=0.0;

	while(pro>pror)
	{
		rsample=curand_uniform(localState_pt)*r_max;
		pror=4.0*rsample*exp(-2.0*rsample);
		pro=curand_uniform(localState_pt)*pro_max;
	}
	*r=rsample;
}										  

__device__ void get_direction(curandState *localState_pt, float *nx, float *ny, float *nz)                                                              					                                                              					
{// uniform sampling on a unit sphere
	float beta = curand_uniform(localState_pt)*2.0f*PI;
	float costheta = 1.0f-2.0f*curand_uniform(localState_pt);
	
	*nx = sqrtf(1-costheta*costheta) * __cosf(beta);
	*ny = sqrtf(1-costheta*costheta)  * __sinf(beta);
	*nz = costheta;
}												  

void PrechemList::initGPUVariables()
{
	
// initial particle info
	printf("total_initial parent particle to go through radiolysis is: %d\n",num_total_paren);

	cudaMemcpyToSymbol(d_num_total, &num_total_paren, sizeof(int), 0, cudaMemcpyHostToDevice);
	
	// branch type and branch model info
	cudaMemcpyToSymbol(d_nbrantype, &nbrantype, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_max_prod_bran, &max_prod_bran, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_num_replace_bran, &num_replace_bran, sizeof(int), 0, cudaMemcpyHostToDevice);
    // parent molecule type and decay branch info for each parent
	cudaMemcpyToSymbol(d_nparentype, &nparentype, sizeof(int), 0, cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(d_max_bran_paren, &max_bran_paren, sizeof(int), 0, cudaMemcpyHostToDevice);
	
    // electron thermalization rms para
    cudaMemcpyToSymbol(d_num_rms_para, &num_para_recom[1], sizeof(int), 0, cudaMemcpyHostToDevice);
  
	printf("finish memory copy for prechemcial stage\n");
}

void PrechemList::run()
{
	//simulating the prechemical stage for the subexcitation electrons: thermalisation or recombination with its parent ionized water
    int nblocks = 1 + (num_total_paren - 1)/NTHREAD_PER_BLOCK_PAR;
    printf("start prechemical run with number of blocks: %d\n", nblocks);
	physiochemical_decay<<<nblocks,NTHREAD_PER_BLOCK_PAR>>>(posx_paren, posy_paren,posz_paren,type_paren,num_bran_paren,branratio_paren,brantype_paren,num_prod_bran,ene_paren,rms_therm_elec,
		para_replace_bran,prodtype_bran);
	//printf("end prechemical run with number of blocks: %d\n", nblocks);
	cudaDeviceSynchronize();	
}

void PrechemList::saveResults()
{
	FILE *fp;
	
	//remove the empty entries or H2O entries from the particle data
	thrust::device_ptr<float> posx_dev_ptr;
	thrust::device_ptr<float> posy_dev_ptr;
	thrust::device_ptr<float> posz_dev_ptr;
	thrust::device_ptr<int> ptype_dev_ptr;
	thrust::device_ptr<int> index_dev_ptr;
	thrust::device_ptr<float> ttime_dev_ptr;
	
	typedef thrust::tuple<thrust::device_vector<int>::iterator, thrust::device_vector<float>::iterator, thrust::device_vector<float>::iterator, thrust::device_vector<float>::iterator, thrust::device_vector<int>::iterator, thrust::device_vector<float>::iterator> IteratorTuple;
        // define a zip iterator
	typedef thrust::zip_iterator<IteratorTuple> ZipIterator;
	
	ZipIterator zip_begin, zip_end, zip_new_end;
	
	ptype_dev_ptr = thrust::device_pointer_cast(&type_paren[0]);		
	posx_dev_ptr = thrust::device_pointer_cast(&posx_paren[0]);	
	posy_dev_ptr = thrust::device_pointer_cast(&posy_paren[0]);	
	posz_dev_ptr = thrust::device_pointer_cast(&posz_paren[0]);	
	index_dev_ptr = thrust::device_pointer_cast(&index_paren[0]);
	ttime_dev_ptr = thrust::device_pointer_cast(&ttime_paren[0]);

	zip_begin = thrust::make_zip_iterator(thrust::make_tuple(ptype_dev_ptr, posx_dev_ptr, posy_dev_ptr, posz_dev_ptr, index_dev_ptr, ttime_dev_ptr));
	zip_end   = zip_begin + num_total_paren * 3;  		
	zip_new_end = thrust::remove_if(zip_begin, zip_end, first_element_equal_255());
	
	cudaDeviceSynchronize();
	
	int	numCurPar = zip_new_end - zip_begin;
		
	printf("After removing, numCurPar = %d\n", numCurPar);
	float *output_posx = (float*)malloc(sizeof(float) * numCurPar);
    float *output_posy = (float*)malloc(sizeof(float) * numCurPar);
    float *output_posz = (float*)malloc(sizeof(float) * numCurPar);
    float *output_ttime = (float*)malloc(sizeof(float) * numCurPar);
    int *output_ptype = (int*)malloc(sizeof(float) * numCurPar);
    int *output_index = (int*)malloc(sizeof(float) * numCurPar);
    
    memcpy(output_posx , posx_paren, sizeof(float)*numCurPar);	
    memcpy(output_posy , posy_paren, sizeof(float)*numCurPar);	
    memcpy(output_posz , posz_paren, sizeof(float)*numCurPar);
    memcpy(output_ptype, type_paren, sizeof(int)*numCurPar);	
    memcpy(output_index, index_paren, sizeof(int)*numCurPar);	
    memcpy(output_ttime, ttime_paren, sizeof(int)*numCurPar);	
	
	std::string fname = document["fileForOutput"].GetString();
	fp = fopen(fname.c_str(), "wb");	
    fwrite(output_posx, sizeof(float), numCurPar, fp);
    fwrite(output_posy, sizeof(float), numCurPar, fp);
	fwrite(output_posz, sizeof(float), numCurPar, fp);
	fwrite(output_ttime, sizeof(float), numCurPar, fp);
	fwrite(output_index, sizeof(int), numCurPar, fp);
	fwrite(output_ptype, sizeof(int), numCurPar, fp);
	fclose(fp);	
	

    cudaFree(num_prod_bran);
	cudaFree(prodtype_bran);
	cudaFree(para_replace_bran);

	cudaFree(num_bran_paren);
	cudaFree(brantype_paren);
	cudaFree(branratio_paren);

	cudaFree(pro_recom);
	cudaFree(rms_therm_elec);

	cudaFree(posx_paren);
	cudaFree(posy_paren);
	cudaFree(posz_paren);
	cudaFree(ene_paren);
	cudaFree(ttime_paren);
	cudaFree(type_paren);
	cudaFree(index_paren);

}
