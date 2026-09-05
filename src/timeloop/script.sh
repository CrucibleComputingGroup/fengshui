#!/bin/bash
# Example Slurm batch script for regenerating the Timeloop database on an HPC
# cluster. This is a TEMPLATE: before submitting, fill in the site-specific
# placeholders below (Slurm account, container image path, and optionally the
# notification settings, which are commented out by default).
#   1. --account=<your-slurm-account>   -> your allocation/charge account
#   2. SINGULARITY_IMAGE                -> path to the Timeloop/Accelergy .sif
# Everything else (resources, array size, workload flags) is portable as-is.
# #SBATCH --mail-type=BEGIN,END        # optional: uncomment and set --mail-user
#SBATCH --job-name=timeloop
#SBATCH --time=3-23:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
# FILL IN: your cluster allocation/charge account (Slurm ignores nothing after the
# value on an #SBATCH line, so keep this comment on its own line).
#SBATCH --account=<your-slurm-account>
#SBATCH --output=%j_%a_timeloop_output.log
#SBATCH --array=0-15%16  # Run 25 jobs, all concurrently

# Load Singularity module
module load singularity

# Set up directories
WORKSPACE_DIR="${WORKSPACE_DIR:-$HOME/timeloop_workspace}"
# FILL IN: path to the Timeloop/Accelergy Singularity image built from
# docker/Dockerfile.timeloop (e.g. singularity build timeloop.sif docker-daemon://...)
SINGULARITY_IMAGE="${SINGULARITY_IMAGE:-$WORKSPACE_DIR/timeloop.sif}"
RESULTS_DIR="$WORKSPACE_DIR/results"
mkdir -p $RESULTS_DIR
mkdir -p $WORKSPACE_DIR/tmp/cacti_data
mkdir -p $WORKSPACE_DIR/neurosim_plugin

# Run your modified Python script
singularity exec \
  -B $WORKSPACE_DIR:/home/workspace \
  -B $WORKSPACE_DIR/tmp/cacti_data:/usr/local/share/accelergy/estimation_plug_ins/accelergy-cacti-plug-in/cacti_inputs_outputs \
  -B $WORKSPACE_DIR/neurosim_plugin:/usr/local/share/accelergy/estimation_plug_ins/accelergy-neurosim-plugin \
  -B $WORKSPACE_DIR/chiplet_timeloop/timeloop/cacti_wrapper.py:/usr/local/share/accelergy/estimation_plug_ins/accelergy-cacti-plug-in/cacti_wrapper.py \
  "$SINGULARITY_IMAGE" \
  python3 /home/workspace/chiplet_timeloop/database_builder.py --networks=gpt_OPT-66B_prefill,gpt_OPT-66B_decode \
  --networks-to-run=gpt_OPT-66B_prefill,gpt_OPT-66B_decode --is-transformer=1 \
  --dram-configs='[{"I":"LPDDR5","O":"LPDDR5"}, {"I":"GDDR7","O":"GDDR7"}, {"I":"HBM3","O":"HBM3"}, {"I":"LPDDR5","O":"GDDR7"}]' \
  --arch=simple_output_stationary \
  --run-id=${SLURM_ARRAY_TASK_ID} \
  --total-runs=16 \
  --skip-softmax=1


echo "Completed run ${SLURM_ARRAY_TASK_ID}"