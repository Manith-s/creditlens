"""MLflow helpers that DON'T crash when no server is running.

In CI / offline runs we still want the pipeline to complete and print metrics.
`mlflow_run(...)` is a context manager: if a tracking server is reachable it logs
normally; otherwise it falls back to a local file store (or a silent no-op) so the
science still runs. For the resume demo, start the real server (`make mlflow`) and
runs show up in the UI with params, metrics, and artifacts.
"""
from __future__ import annotations

import contextlib
from typing import Any

from src.config import CFG


@contextlib.contextmanager
def mlflow_run(experiment: str, run_name: str, tracking_uri: str | None = None):
    """Yield an object with .log_params/.log_metrics/.log_artifact/.sklearn_log.

    Tries the configured server; on failure, falls back to a local ./mlruns file
    store; if even that import fails, yields a no-op recorder.
    """
    uri = tracking_uri or CFG.mlflow_tracking_uri
    try:
        import mlflow

        try:
            mlflow.set_tracking_uri(uri)
            mlflow.set_experiment(experiment)
            mlflow.start_run(run_name=run_name)
        except Exception:  # server unreachable -> local file store
            mlflow.set_tracking_uri(f"file://{(CFG.root / 'mlruns').as_posix()}")
            mlflow.set_experiment(experiment)
            mlflow.start_run(run_name=run_name)
        try:
            yield _MlflowRecorder(mlflow)
        finally:
            mlflow.end_run()
    except Exception:  # mlflow not installed at all
        yield _NoopRecorder()


class _MlflowRecorder:
    def __init__(self, mlflow):
        self._mlflow = mlflow

    def log_params(self, params: dict[str, Any]) -> None:
        self._mlflow.log_params({k: str(v) for k, v in params.items()})

    def log_metrics(self, metrics: dict[str, float], prefix: str = "") -> None:
        clean = {f"{prefix}{k}": float(v) for k, v in metrics.items()}
        self._mlflow.log_metrics(clean)

    def log_artifact(self, path: str) -> None:
        with contextlib.suppress(Exception):
            self._mlflow.log_artifact(path)

    def log_text(self, text: str, artifact_file: str) -> None:
        with contextlib.suppress(Exception):
            self._mlflow.log_text(text, artifact_file)

    def sklearn_log(self, model, name: str, registered_model_name: str | None = None):
        with contextlib.suppress(Exception):
            self._mlflow.sklearn.log_model(
                model, name=name, registered_model_name=registered_model_name
            )


class _NoopRecorder:
    def log_params(self, *a, **k):
        pass

    def log_metrics(self, *a, **k):
        pass

    def log_artifact(self, *a, **k):
        pass

    def log_text(self, *a, **k):
        pass

    def sklearn_log(self, *a, **k):
        pass
