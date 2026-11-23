#!/bin/bash
#SBATCH --job-name=mhd2d
#SBATCH --nodes=1
#SBATCH --ntasks=1                # mhd2d is OpenMP (shared memory), not MPI: ONE process
#SBATCH --cpus-per-task=32        # = OpenMP threads; 1024x512 scales to ~16-32 threads
#SBATCH --mem-per-cpu=4591
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%x-%j.out

# Usage (from the repo root):
#   sbatch scripts/run_slurm.sh                      # uses config/ctrl.txt
#   sbatch scripts/run_slurm.sh config/my_run.txt    # any other deck
#   sbatch --cpus-per-task=64 scripts/run_slurm.sh   # change thread count
# To continue a test-31 run after a timeout: set `resume = .true.` in the deck
# and resubmit (checkpoints are written every ckpt_dt).

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

CTRL="${1:-config/ctrl.txt}"
[ -f "$CTRL" ] || { echo "control deck not found: $CTRL"; exit 1; }

module purge
module load GCC/14.3.0

# Compile on the compute node: the Makefile uses -march=native.
# flock stops simultaneous jobs from building into build/ at the same time.
mkdir -p build
flock build/.lock make -j"${SLURM_CPUS_PER_TASK}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

OUT="$(build/mhd2d --print-outdir "$CTRL")"
echo "job=${SLURM_JOB_ID} host=$(hostname) deck=$CTRL threads=$OMP_NUM_THREADS output=$OUT"

set +e
srun --cpus-per-task="${SLURM_CPUS_PER_TASK}" build/mhd2d "$CTRL"
RC=$?
set -e
# Keep the Slurm log next to the results
cp "slurm-${SLURM_JOB_NAME}-${SLURM_JOB_ID}.out" "$OUT/" 2>/dev/null || true
exit $RC
