#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --nodes={nodes}
#SBATCH --ntasks-per-node={cores_cpu}

cd "{workdir}"
{launch_command}
