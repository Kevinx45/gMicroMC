# gMicroMC_v2.0
Updated by Youfang Lai. Author email: youfanglai@gmail.com  
Correpondance: xun.jia@utsouthwestern.edu and yujie.chi@uta.edu  

Every researchers is welcome to modify and distribute this package with no commercial purpose. The authors guarantee that all developped modules have been uploaded, but claim no
responsibility for the results produced by users.
If you want to cite this work, please cite
* initial development for electron  
Tsai M, Tian Z, Qin N, Yan C, Lai Y, Hung S-H, Chi Y and Jia X 2020 A new open-source GPU-based microscopic Monte Carlo simulation tool for the calculations of DNA damages caused by ionizing radiation --- Part I: Core algorithm and validation Med. Phys. 47 1958-70
* oxygen module  
Lai Y, Jia X and Chi Y 2020 Modeling the effect of oxygen on the chemical stage of water radiolysis using GPU-based microscopic Monte Carlo simulations, with an application in FLASH radiotherapy Phys. Med. Biol. 
* proton and concurrent method  
To be submitted

# Updated features
May 3rd, 2021  
1. updated gMicroMC package with more comments and smoother control of the output by using class
2. all fuctions were divided into two types, kernel functions executed by GPU in \*.cu files and typical \*.cpp files

# Overview about microscopic simulation
To make full use of this package and get meaningful results, users should have general picture about what we want and what the package can provide. 
## 1）what we want
The microscopic simulation tries to acquire the information about physics track and subsequent waterradiolysis process, the so-called track structure, so that we can know how DNA is damaged by defining the damage format from physical energy deposition and radical attack. The track structure can span multi time scale from 10<sup>-15</sup> s to 10<sup>-5</sup> s. Hence, the simulation of track structure is usually divided into three stages -- physical stage covering 10<sup>-15</sup> s to 10<sup>-12</sup> s for energy deposition positions and initial ionized or excited water molecules, physicochemical stage (sometimes referred to prechemical stage) giving the initial position and types of radicals (molecules) to participate the following chemical stage, which deals with the diffusion and mutual ractions covering time scale from 10<sup>-12</sup> s to 10<sup>-6</sup> s. Here, we assumed the physicochemical stage is very short and all the radicals from the same primary particles were produced at 10<sup>-12</sup> s. In physical stage, we sampled step length and interaction types according to predefined cross section tables. In physicochemical stage, we de-excite the water molecules through a predefine branch probabilities. In chemical stage, we diffused radicals and then check their mutual reactions step by step. DNA may intefere the radicals in chemical stage and it is called concurrent method in our package. Finally, all the physical events and radical attack produce DNA damage through a predefined way, for example, local energy deposition exceeding 17.5 eV. Typically, we are more interested in Double Strand Break (DSB), or at least closely connected Single Strand Break (SSB). The reason is that such damages are more lethal to cells while isolated strand break could be repaired by the DNA repair mechanism. The repair process is beyond the scope of this package. The recording DNA damage pattern is required to be analysed in 10<sup>-1</sup> nm scale while DNA is highly folded in a space with dimension around 10<sup>4</sup> nm. Hence, the DNA geometry is stored as multiscale structure. Interested users are excouraged to read the following papers and the references therein.
1. Friedland W, Dingfelder M, Kundrát P and Jacob P 2011 Track structures , DNA targets and radiation effects in the biophysical Monte Carlo simulation code PARTRAC Mutation Research - Fundamental and Molecular Mechanisms of Mutagenesis 711 28-40
2. Nikjoo H, Emfietzoglou D, Liamsuwan T, Taleei R, Liljequist D and Uehara S 2016 Radiation track , DNA damage and response — a review Reports on Progress in Physics 79 116601
3. Tsai M, Tian Z, Qin N, Yan C, Lai Y, Hung S-H, Chi Y and Jia X 2020 A new open-source GPU-based microscopic Monte Carlo simulation tool for the calculations of DNA damages caused by ionizing radiation --- Part I: Core algorithm and validation Med. Phys. 47 1958-70

## 2）what the package can provide
The advantage of using gMicroMC is making full use of GPU to accelerate the simulation process. It is very important because the computational efficiency has been a bottleneck for increasing the simulation accuracy and removing unnecessary limits, for example, treating oxygen explicitly as molecules in the early age of chemical stage rather than viewing them as continuum background. The package does not introduce new physical or chemical interpretation. Hence, what gMicroMC can provide is basically the same as other CPU packages:
- Deposited energy, positions, track index etc. in the physics stage. (Check Data structure)
- Initial types of radicals and their distributions. (Check output from prechemical stage)
- Yields of different radicals at different moments, DNA damage sites in chemical stage.
- DNA strand break pattern after DNA damage grouping.
***

# Usage
## Structure
The code is structured as  
./ root folder  
&nbsp;&nbsp;&nbsp;&nbsp;./src/ --> source code  
&nbsp;&nbsp;&nbsp;&nbsp;./inc/ --> header files  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;./inc/rapidjson --> rapidjson library to deal with json files  
&nbsp;&nbsp;&nbsp;&nbsp;./tables/ --> predefined data for different process  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;./tables/physics --> physics cross sections  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;./tables/prechem --> info for decay channes and recombination of hydrated electrons  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;./tables/chem --> infor for species and their reactions  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;./tables/dna --> stored dna structure in multiscale  
&nbsp;&nbsp;&nbsp;&nbsp;./output/ --> output directory  
&nbsp;&nbsp;&nbsp;&nbsp;./examples/ --> list of config files for different settings  
## Compile
nvcc main.cpp ./src/* -o gMicroMC -I ./inc -rdc=true  
A make file will be provided later for easier compile  

## Run the program
After compilation, the users needs to provide correct config.txt file, where illustration of parameters have been listed.  
Then running ./gMicroMC command will give you ourput files as defined in the config.txt.  

## Example
User can define their own logic to run the program for different senerios. Currently, main.cpp provides a scheme to calculate DNA damage for a single particle.  

The work flow is  
prepare arrays like random seeds  
--> generating initial particles physicsList::h_particles  
--> simulate physics stage physicsList::run()  
--> store initial positions and types of water molecules physicsList::saveResults()  
--> reading prechemical stage PrechemList::readWaterStates()  
--> simulate prechemical stage and save results PrechemList::run()  
-->  reading chemical stage ChemList::d_posx, ChemList::d_posy, ChemList::d_posz, ChemList::d_ttime, ChemList::d_ptype, ChemList::d_index  
--> run chemical stage ChemList::run()  
--> save results ChemList::saveResults()  
--> read in for DNA damage analysis DNAList::posx  
--> simulated DNA damage analysis DNAList::run()  
--> DNA damage summary DNAList::saveResults();

All information is given in config.txt file, which is in json format.

Note
1. the saveResults() and readIn functions are not required mandatorily. Users can directly set the values in correspoding host arrays. For exampls, instead of physicsList::saveResults() and then prechemicalList::ReadIn(), users can directly set values of prechemicalList::posx, prechemicalList::posy, prechemicalList::posz, prechemicalList::ptype, prechemicalList::ttime, prechemicalList::index. This can be done in main.cpp fuction.
2. The reason why saveResults() fuction exists is due to the concern of memory. Saving into files and then either read in by batchs or apply constraints to reduce the number of events (radicals) is safer.
3. The users has total freedom to change the data in ./tables folder, which alter the defined physics interaction or decay channel. So be careful. 

