#!/bin/bash -l
#SBATCH --job-name={job_name}
#SBATCH --output=out.%j
#SBATCH --error=err.%j
#SBATCH --partition=normal
#SBATCH --account=sait
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={cores_cpu}
#SBATCH --cpus-per-task=1
#SBATCH --exclusive

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

mpirun -np {cores_cpu} {command}

echo "Simulation finished at $(date)"
