"""
Migrates all MLflow runs from the file store (notebooks/mlruns/)
to the SQLite backend (data/mlflow.db).
"""
import mlflow
from mlflow.tracking import MlflowClient

SRC  = "file:///app/notebooks/mlruns"
DEST = "sqlite:////app/data/mlflow.db"

src_client  = MlflowClient(tracking_uri=SRC)
dest_client = MlflowClient(tracking_uri=DEST)

experiments = src_client.search_experiments()
print(f"Found {len(experiments)} experiment(s) to migrate.")

for exp in experiments:
    if exp.name == "Default":
        continue

    # Get or create experiment in destination
    existing = dest_client.get_experiment_by_name(exp.name)
    if existing:
        dest_exp_id = existing.experiment_id
    else:
        dest_exp_id = dest_client.create_experiment(exp.name)
    print(f"\nExperiment: {exp.name} → dest id {dest_exp_id}")

    runs = src_client.search_runs(experiment_ids=[exp.experiment_id], max_results=5000)
    print(f"  {len(runs)} run(s) to migrate...")

    for run in runs:
        r = run.info
        d = run.data

        dest_run = dest_client.create_run(
            experiment_id=dest_exp_id,
            run_name=r.run_name,
            start_time=r.start_time,
            tags=d.tags,
        )
        dest_run_id = dest_run.info.run_id

        # Log params
        for k, v in d.params.items():
            dest_client.log_param(dest_run_id, k, v)

        # Log metrics
        for k, v in d.metrics.items():
            dest_client.log_metric(dest_run_id, k, v)

        # Set end time and status
        dest_client.set_terminated(
            dest_run_id,
            status=r.status,
            end_time=r.end_time,
        )

    print(f"  Done.")

print("\nMigration complete.")
