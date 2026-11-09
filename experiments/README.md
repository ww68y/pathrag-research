# Evaluation scripts

This directory contains source scripts for development experiments and evaluation workflows.

## Before running a script

- Read the script and its argument parser first. Some runners can call hosted model APIs, consume quota, and incur charges.
- Prepare the data and model credentials required by that specific script. No evaluation datasets or result files are included in this repository.
- Keep datasets, caches, logs, and generated artifacts outside version control.
- Record the code revision, model/provider version, prompt/configuration, and input-data version when reporting results.

## Naming guide

The file prefixes are conventions rather than a complete registry:

- `run_*`: execute an experiment or model-backed workflow
- `build_*`: prepare derived inputs or evaluation assets
- `evaluate_*` / `audit_*`: score or inspect saved outputs
- `summarize_*` / `analyze_*`: aggregate or diagnose existing records
- `verify_*` / `check_*`: check consistency or execution state

Most scripts are research utilities rather than stable command-line interfaces. Their parameters and required inputs may change between versions.
