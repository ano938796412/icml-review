#!/bin/bash

#SBATCH --job-name=arbitrary

#SBATCH --array=1-480

#SBATCH --partition=$NODE

#SBATCH --requeue

#SBATCH --nodes=1

#SBATCH --exclude=$EXCLUDE

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=10

#SBATCH --gres=gpu:1

#SBATCH --mem=20000

#SBATCH --time=5:00:00

#SBATCH --output=$PROJECT_DIR/obfuscation/slurm/logs/%x_%A_%a_%N.txt

#SBATCH --error=$PROJECT_DIR/obfuscation/slurm/logs/%x_%A_%a_%N_err.txt

ulimit -n 4096 # max limit allowed on slurm

SCRIPT_DIR="${SLURM_JOB_NAME}" # as stated in generate_scripts.sh

activate_env() {
  source $USER_DIR/miniconda3/etc/profile.d/conda.sh && conda activate "$1"
}

DB_ENV=db
TORCH_ENV=coco

echo "Running mongod with env ${DB_ENV}"
activate_env "${DB_ENV}"

LOG_DIR=$USER_DIR/fiftyone/${SCRIPT_DIR}/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "${LOG_DIR}"
echo LOG_DIR: "${LOG_DIR}"

RND_PT=$(shuf -i 1025-65000 -n 1) # 1-1024 requires sudo
echo RND_PT: "${RND_PT}"

# https://www.mongodb.com/docs/manual/reference/program/mongod/
# https://www.mongodb.com/docs/manual/administration/configuration/
# https://www.mongodb.com/docs/manual/tutorial/manage-mongodb-processes/
# https://www.mongodb.com/docs/manual/reference/configuration-file-settings-command-line-options-mapping/#std-label-conf-file-command-line-mapping
mongod --dbpath "${LOG_DIR}" --fork \
  --logpath "${LOG_DIR}".log \
  --port "${RND_PT}" || exit

# https://voxel51.com/docs/fiftyone/user_guide/config.html#using-a-different-mongodb-version
mongo --eval 'db.adminCommand({setFeatureCompatibilityVersion: "4.4"})' --port "${RND_PT}"

echo
echo "Running pytorch with env ${TORCH_ENV}"
activate_env "${TORCH_ENV}"

# https://voxel51.com/docs/fiftyone/user_guide/config.html#configuring-a-mongodb-connection
export FIFTYONE_DATABASE_URI=mongodb://127.0.0.1:${RND_PT}/
export FIFTYONE_SHOW_PROGRESS_BARS=false

export NUMEXPR_MAX_THREADS=$((SLURM_CPUS_PER_TASK - 2))
export OMP_NUM_THREADS=$((SLURM_CPUS_PER_TASK - 2))

echo NUMEXPR_MAX_THREADS: "${NUMEXPR_MAX_THREADS}"
echo OMP_NUM_THREADS: "${OMP_NUM_THREADS}"

# python working directory is `obfuscation`
cd $PROJECT_DIR/obfuscation || exit
bash "./slurm/${SCRIPT_DIR}/${SLURM_ARRAY_TASK_ID}.sh"

echo
echo "Completed and shutting down mongod..."
activate_env "${DB_ENV}"
mongod --shutdown --dbpath "${LOG_DIR}"

sacct --format=JobID,Elapsed,NodeList,Partition,State,MaxDiskRead,MaxDiskWrite,MaxRSS,MaxVMSize --units=G -j "${SLURM_JOB_ID}"
