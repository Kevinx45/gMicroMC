#ifndef PRECHEMICAL_H
#define PRECHEMICAL_H
#include "global.h"

#define MAXNUMBRANCHPROD 3 // maximum number of products a branch can have
#define MAXBRANCHTYPE 6
#define MAXNUMBRANCH 3

#define PBRANCH2RECOMB 0.3 //the recombined H2O* deexcited to be H2O
#define PBRANCH11RECOMB 0.55 //0.55 // the recombined H2O* dissociative deexcited to be H. + OH.
#define PBRANCH12RECOMB 0.15 //0.15 // the recombined H2O* dissociative deexcited to be H2 + OH. + OH.
#define NTHREAD_PER_BLOCK_PAR 512


class PrechemList
{
public:
	PrechemList();
	~PrechemList();

	void readBranchInfo(std::string fname);
	void readThermRecombInfo(std::string fname); // loading the thermalization mean distance and recombination probability of the subexcitation electrons for prechemical stage simulation
	void readWaterStates();

	void initGPUVariables();
	void run();
	void saveResults();
// parameters for Branch Infor
	int nbrantype; // number of all the different branch types
  int max_prod_bran; // maximum products from one branch
  int *num_prod_bran; // the number of the products for different branch types
	int *prodtype_bran; // the species type of the products for different branch types
  int num_replace_bran; // number of parameters in the product hoping model (1 rms of hole hopping, two rms and coefficient for each product (1+2+2*3=9 entries for each branch) )
	float *para_replace_bran; // for each branch type, all the info 
 
  int nparentype; // number of parent molecular types
  int max_bran_paren; // maximum branches from one parent molecule
  int *num_bran_paren; // the number of the branches for different parent molecules
  int *brantype_paren; // the type of different branches for different parent molecules
  float *branratio_paren; // the ratio of different branches for different parent molecules
	
	// parameters for thermal relaxation
   float Ecut_recom; // cutoff energy for the electron-hole recombination in eV and number of paramters for the following three variables
   int num_para_recom[2];
   float *pro_recom; // recombination probability , 7 parameters
   float *rms_therm_elec; //product replacement rms after recombination in nm, 7 parameters


// parameters for information of waterradiolysis state
	int num_total_paren; // total number of the electrons and water molecules
	float *posx_paren, *posy_paren, *posz_paren; //current positions of the molecules
	float *ene_paren, *ttime_paren; // energy of the molecules
	int *type_paren; // the ionization state or excitation state of the water molecules (-1 for electrons)
	int *index_paren;
   
  float *electron_container,*ion_container;
  int *ion_tag, *ion_index, *elec_index;
  
	
	
// GPU variables
	cudaStream_t stream[5];
	float *d_posx, *d_posy, *d_posz; // the GPU variables to store the positions of the particles (a larger memory is required to include the product of prechemical stage) 
	float *d_ene_paren, *d_ttime; // initial energies of the initial particles
	int *d_ptype, *d_index; // the species type of the particles (255 for empty entries or produced H2O)	

	//int d_num_total; 
	//int d_nbrantype;
	//int d_max_prod_bran; 
	//int d_num_replace_bran;
	int *d_num_prod_bran, *d_prodtype_bran;
	float *d_para_replace_bran;

	//int d_nparentype;
	//int d_max_bran_paren; 
	int *d_num_bran_paren, *d_brantype_paren;
	float *d_branratio_paren;

	//int d_num_rms_para;
	float *d_rms_therm_elec;


};

#endif