"""Several GPUs and several nodes: one process per node, every card of the node to it, joined by `jax.distributed`.

A Slurm job runs `srun` with one task per node; each task calls `initialize()` before anything asks jax for a
device. quantax's `("process", "device")` mesh then covers every card of every node: the chains, the forward and
backward passes and the Jacobian are sharded over all of them, and the SR solve's reductions run across nodes.
Outside such a step (one task, or no Slurm) `initialize()` does nothing, and a single process drives every card it
sees. Only process 0 prints and writes.

"""
from __future__ import annotations

import os
import sys

__all__ = ["slurm_step_tasks", "local_device_ids", "initialize", "is_writer"]


def slurm_step_tasks(environ=None):
    """How many processes the enclosing `srun` step started (`SLURM_STEP_NUM_TASKS`); 1 outside one. Not
    `SLURM_NTASKS`, which is the allocation's: a batch script calling Python without `srun` would wait forever in
    `initialize` for a partner never started."""
    environ = os.environ if environ is None else environ
    return int(environ.get("SLURM_STEP_NUM_TASKS", "1") or 1)


def local_device_ids(environ=None):
    """Every card this process sees (`CUDA_VISIBLE_DEVICES`, renumbered from 0), for `jax.distributed.initialize`:
    left to itself it gives a Slurm task ONE card (that of `SLURM_LOCALID`). `None` when the variable is unset."""
    environ = os.environ if environ is None else environ
    visible = environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible.strip():
        return None
    return list(range(len(visible.split(","))))


def initialize(quiet_followers=True):
    """Join the processes of a multi-task `srun` step (no-op otherwise); silence stdout on every process but 0."""
    import jax
    if slurm_step_tasks() > 1:
        jax.distributed.initialize(local_device_ids=local_device_ids())
        if quiet_followers and jax.process_index() != 0:
            sys.stdout = open(os.devnull, "w")
    return jax.process_index(), jax.process_count()


def is_writer():
    import jax
    return jax.process_index() == 0
