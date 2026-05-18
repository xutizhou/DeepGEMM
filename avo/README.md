# AVO DGX Spark Dense GEMM Workflow

This branch is the AVO working branch for DGX Spark dense GEMM optimization.

Canonical branch:

```bash
avo/dgx-spark-dense-gemm
```

Local worktree:

```bash
/Users/xutingz/workspace/gitsrc/DeepGEMM-avo-dgx-spark
```

DGX Spark checkout:

```bash
/home/cecily/avo-dgx-spark/DeepGEMM
```

Remote fork:

```bash
git@github.com:xutizhou/DeepGEMM.git
```

Use git as the transport between local editing and DGX Spark runs:

```bash
# Local
git add <files>
git commit -m "..."
git push xutizhou avo/dgx-spark-dense-gemm

# DGX Spark
git pull --ff-only origin avo/dgx-spark-dense-gemm
```

Run AVO from the AVO repo, with this DeepGEMM checkout as `--working-dir`.
