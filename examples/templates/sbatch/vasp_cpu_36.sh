#!/bin/bash -l
#SBATCH --job-name={job_name}
#SBATCH --output=out.%j
#SBATCH --error=err.%j
#SBATCH --partition=normal
#SBATCH --account=sait
#SBATCH --nodes=1
#SBATCH --ntasks=36
#SBATCH --cpus-per-task=1
#SBATCH --exclusive

set -euo pipefail
cd "{workdir}"

module purge
source /home/a4s5d/software/VASP/CPU_vasp.6.6.0/ENV

export OMPI_MCA_orte_tmpdir_base=/dev/shm
export TMPDIR=/dev/shm
export OMP_NUM_THREADS=1
ulimit -s unlimited

echo "--------------------------------"
echo "Job started at $(date)"
echo "Running on node: $SLURM_NODELIST"
echo "Using total cores: $SLURM_NTASKS"
echo "Working directory: $(pwd)"
echo "--------------------------------"

mpirun {command}

echo "Simulation finished at $(date)"
