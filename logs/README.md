# logs

SLURM job logs from HPC cluster runs. Each job produces two files:

```
<job_id>.out   — standard output
<job_id>.err   — standard error / warnings
```

These are generated automatically when submitting jobs via `batch.sh`. Check `.err` files first when diagnosing failed runs.
