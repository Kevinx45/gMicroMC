#ifndef PRECHEMICAL_CUH
#define PRECHEMICAL_CUH

#include "global.h"
#include <thrust/device_vector.h>

struct first_element_equal_255
{
  __host__ __device__
  bool operator()(const thrust::tuple<const int&, const float&, const float&, const float&, const int&, const float&> &t)
  {
      return thrust::get<0>(t) == 255;
  }
};

__global__ void physiochemical_decay(float *d_posx, // x position of the particles (input and output)
                                    float *d_posy,
									float *d_posz,
									int *d_ptype,
									//int d_num_total,
									int *d_num_bran_paren,
									float *d_ratio_bran_paren,
									int *d_brantype_paren,
									int *d_num_prod_bran,
									float *d_ene_paren,
									//int d_num_rms_para,
									float *d_rms_therm_elec,
									//int d_max_prod_bran,
									//int d_max_bran_paren,
									float *d_para_replace_bran,
									int *d_prodtype_bran
									);

__device__ void get_distance(curandState *localState_pt, float rms, float *r,float ene,int flag);

__device__ void get_electron_distance(curandState *localState_pt, float *r);  

__device__ void get_direction(curandState *localState_pt, float *nx, float *ny, float *nz) ;                                                      	                                                                                                                                                                                                                                 	                                                                                                                                       


#endif