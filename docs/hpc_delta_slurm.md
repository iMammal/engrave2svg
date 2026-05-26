# NCSA Delta SLURM Notes

`engrave2svg` sensitivity runs are CPU-first and deterministic. They do not require CUDA, GPU nodes, or GPU libraries.

The local batch command creates one independent trial directory per parameter set:

```bash
python engrave2svg.py examples/synthetic_panel.png \
  --sensitivity \
  --sensitivity-dir sensitivity_delta \
  --jobs 1
```

Each trial writes:

- `trials/trial_0000/params.json`
- `trials/trial_0000/command.txt`
- `trials/trial_0000/output.svg`
- `trials/trial_0000/debug/*.png`

The `command.txt` files are intentionally standalone so they can be dispatched later as SLURM array tasks.

## Prepare the Environment

Create the virtual environment once before submitting the array job:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Example Array Script

Save as `run_delta_array.sbatch` from the project root after generating `sensitivity_manifest.csv` locally or in a Delta login session.

```bash
#!/bin/bash
#SBATCH --job-name=engrave2svg-sensitivity
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --array=0-11
#SBATCH --output=slurm-%A_%a.out

set -euo pipefail

module purge
module load python

cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate

COMMAND_FILE=$(printf "sensitivity_delta/trials/trial_%04d/command.txt" "$SLURM_ARRAY_TASK_ID")
bash "$COMMAND_FILE"
```

Notes:

- Use the CPU partition unless your site policy says otherwise.
- Keep `--jobs 1` for array tasks; SLURM provides the parallelism.
- No CUDA modules are needed for this branch.
- A later `hpc` branch can add scheduler helpers, but this branch only prepares standalone trial outputs.
