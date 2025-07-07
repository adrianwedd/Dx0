# Dataset Versioning with DVC

This project uses [DVC](https://dvc.org/) to track clinical case data and generated embeddings.

## Installation

Install DVC using pip along with the optional S3 extras if you plan to use cloud storage:

```bash
pip install "dvc[s3]"
```

## Initialising Data

After cloning the repository, run the following command to download the
tracked datasets from the configured remote (``dvc_storage`` by default):

```bash
dvc pull
```

This will populate the ``data/raw_cases`` and ``data/sdbench/cases`` directories
as well as the ``data/embeddings`` folder.

## Contributing Data

If you add or modify files under ``data/`` run:

```bash
dvc add data/raw_cases data/sdbench/cases data/embeddings
```

Commit the resulting ``*.dvc`` files and push the updated data to the remote:

```bash
dvc push
```

This ensures that large datasets remain versioned separately from the Git history
while remaining reproducible for all contributors.
