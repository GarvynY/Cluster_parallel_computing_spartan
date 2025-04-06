# Assignment 1 :parallel computing on spartan HPC cluster
## Report Structure
1. Scripts for submitting the job on Spartan
2. The approach to parallize the code
3. Description of performance of nodes and cores
4. The results tables for three options
5. Relate results to Amdahl's law describe the potential performance

## Project Structure

+ bash_scripts -- bash shell scripts used in the project
+ data -- small size test data
+ output -- results
+ slurm_scripts -- slurm scripts used for three ways
+ src
  + scripts_on_spartan -- Parallelized python code running on spartan
    + mpi_parallel_spartan -- 1st version(work on the 106k and 16m dataset)
    + mastodon_analysis -- final version(work on the 144G dataset)
  + test_scripts -- some try in the mid
+ docx file -- report