#!/bin/bash
# The data's GPU checks, in its own environment (no other code on PYTHONPATH): a two-sweep measurement smoke from the
# state's stored end chains (no burn-in), then the tests under the production XLA flags.
#   cd data && uv sync --frozen --extra test && bash tests/run_gpu_checks.sh [state]
set -uo pipefail
STATE=${1:-proj16x4-t0-nv4}
PROD="--xla_gpu_shard_autotuning=false --xla_gpu_experimental_enable_fusion_autotuner=false --xla_gpu_enable_triton_gemm=false --xla_gpu_enable_command_buffer="
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
unset PYTHONPATH
cd "$(dirname "$0")/.."
t0=$SECONDS
echo "=== nngtn-measure smoke ($STATE: 2 sweeps of 256 samples from its end chains)"
XLA_FLAGS="$PROD" .venv/bin/nngtn-measure --config measurements/"$STATE"/config.toml --out runs/smoke/"$STATE" \
    --chains-in weights/"$STATE"_spins.npz --thermal 0 --nsamples 256 --nsweeps 2 --save-every 1 2>&1 \
    | grep -a "sweep\|E_per_site\|Traceback\|Error"
echo "=== exit ${PIPESTATUS[0]} ($((SECONDS - t0)) s)"
echo "=== pytest, production flags"
XLA_FLAGS="$PROD" .venv/bin/python -m pytest -q -rs tests 2>&1 | tail -6
echo "=== all done ($((SECONDS - t0)) s)"
