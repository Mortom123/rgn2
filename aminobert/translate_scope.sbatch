#!/bin/bash
#SBATCH -A p_drug
#SBATCH --job-name=aminobert_convert_scope
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/s0794732/logs/%j.log
#SBATCH --error=/home/s0794732/logs/%j.err
#SBATCH --mem-per-cpu=2048M
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mail-user martin.menzel1@mailbox.tu-dresden.de 	### tell the batch system your email address to get updates about the jobs status
#SBATCH --mail-type ALL ### specify for what type of events you want to get a mail; valid options beside ALL are: BEGIN, END, FAIL, REQUEUE
#SBATCH --gres=gpu:2
#SBATCH --partition=alpha

module purge
set -e

export IMAGE=/home/h1/s0794732/images/aminobert.sif
export WD=/home/h1/s0794732/code/rgn2/aminobert

cd ${WD}

singularity exec --nv -C -e \
-B ${WD}:${WD} \
-B /beegfs/.global1/ws/s0794732-aminobert/embeddings \
--pwd ${WD} \
${IMAGE} \
python3 translate_scope.py

echo "Finished at `date`"
echo "__DONE__"