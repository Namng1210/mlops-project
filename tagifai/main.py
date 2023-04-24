import json
import tempfile
import warnings
from argparse import Namespace
from pathlib import Path

import joblib
import mlflow
import optuna
import pandas as pd
import typer
from numpyencoder import NumpyEncoder
from optuna.integration.mlflow import MLflowCallback

from config import config
from config.config import logger
from tagifai import data, predict, train, utils

warnings.filterwarnings("ignore")

app = typer.Typer()


@app.command()
def etl_data():
    """Extract, load and transform our data assets"""
    # Extract and load
    projects = pd.read_csv(config.PROJECTS_URL)
    tags = pd.read_csv(config.TAGS_URL)
    projects.to_csv(Path(config.DATA_DIR, "projects.csv"), index=False)
    tags.to_csv(Path(config.DATA_DIR, "tags.csv"), index=False)

    # Transform
    df = pd.merge(projects, tags, on="id")
    df = df[df.tag.notnull()]  # drop rows with no tag
    df.to_csv(Path(config.DATA_DIR, "labeled_projects.csv"), index=False)

    logger.info("✅ Saved data!")


@app.command()
def train_model(
    args_fp: str = "config/args.json", experiment_name: str = "baselines", run_name: str = "sgd"
) -> None:
    """Train a model given arguments"""
    # Load labeled data
    df = pd.read_csv(Path(config.DATA_DIR, "labeled_projects.csv"))

    # Train
    args = Namespace(**utils.load_dict(filepath=args_fp))
    mlflow.set_experiment(experiment_name=experiment_name)
    with mlflow.start_run(run_name=run_name):
        run_id = mlflow.active_run().info.run_id
        artifacts = train.train(df=df, args=args)

        # Log metrics and performance
        performance = artifacts["performance"]
        print(json.dumps(performance, indent=2))
        mlflow.log_metrics({"precision": performance["overall"]["precision"]})
        mlflow.log_metrics({"recall": performance["overall"]["recall"]})
        mlflow.log_metrics({"f1": performance["overall"]["f1"]})
        mlflow.log_params(vars(artifacts["args"]))

        # Log artifacts
        with tempfile.TemporaryDirectory() as dp:
            utils.save_dict(vars(artifacts["args"]), Path(dp, "args.json"), cls=NumpyEncoder)
            artifacts["label_encoder"].save(Path(dp, "label_encoder.json"))
            joblib.dump(artifacts["vectorizer"], Path(dp, "vectorizer.pkl"))
            joblib.dump(artifacts["model"], Path(dp, "model.pkl"))
            utils.save_dict(performance, Path(dp, "performance.json"))
            mlflow.log_artifacts(dp)

    # Save to config
    open(Path(config.CONFIG_DIR, "run_id.txt"), "w").write(run_id)
    utils.save_dict(performance, Path(config.CONFIG_DIR, "performance.json"))


@app.command()
def optimize(
    args_fp: str = "config/args.json", study_name: str = "optimization", num_trials: int = 20
) -> None:
    """Optimize hyperparameters."""
    # Load labeled data
    df = pd.read_csv(Path(config.DATA_DIR, "labeled_projects.csv"))

    # Optimize
    args = Namespace(**utils.load_dict(filepath=args_fp))
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
    study = optuna.create_study(study_name=study_name, direction="maximize", pruner=pruner)
    mlflow_callback = MLflowCallback(tracking_uri=mlflow.get_tracking_uri(), metric_name="f1")
    study.optimize(
        lambda trial: train.objective(args, df, trial),
        n_trials=num_trials,
        callbacks=[mlflow_callback],
    )

    # Best trial
    trials_df = study.trials_dataframe()
    trials_df = trials_df.sort_values(["user_attrs_f1"], ascending=False)
    args = {**args.__dict__, **study.best_trial.params}
    utils.save_dict(d=args, args_fp=args_fp, cls=NumpyEncoder)
    print(f"\nBest value (f1): {study.best_trial.value}")
    print(f"Best hyperparameters: {json.dumps(study.best_trial.params, indent=2)}")


@app.command()
def load_artifacts(run_id: str = None):
    """Load artifacts for a given run_id.
    Args:
        run_id (str): id of run to load artifacts from.
    Returns:
        Dict: run's artifacts.
    """
    if not run_id:
        run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()

    # Locate specifics artifacts directory
    experiment = mlflow.get_run(run_id=run_id).info.experiment_id
    artifact_dir = Path(config.MODEL_REGISTRY, experiment, run_id, "artifacts")

    # Load objects from run
    args = Namespace(**utils.load_dict(filepath=Path(artifact_dir, "args.json")))
    vectorizer = joblib.load(Path(artifact_dir, "vectorizer.pkl"))
    label_encoder = data.LabelEncoder.load(fp=Path(artifact_dir, "label_encoder.json"))
    model = joblib.load(Path(artifact_dir, "model.pkl"))
    performance = utils.load_dict(filepath=Path(artifact_dir, "performance.json"))

    return {
        "args": args,
        "label_encoder": label_encoder,
        "vectorizer": vectorizer,
        "model": model,
        "performance": performance,
    }


@app.command()
def predict_tag(text: str = "", run_id: str = None):
    """Predict tag for text.

    Args:
        text (str): input text to predict label for.
        run_id (str, optional): run id to load artifacts for prediction. Defaults to None.
    """
    if not run_id:
        run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
    artifacts = load_artifacts(run_id=run_id)
    prediction = predict.predict(texts=[text], artifacts=artifacts)
    logger.info(json.dumps(prediction, indent=2))
    return prediction


if __name__ == "__main__":
    app()

# if __name__ == "__main__":
#     etl_data()
# args_fp = Path(config.CONFIG_DIR, "args.json")
# # train_model(args_fp, experiment_name="baselines", run_name="sgd")
# # optimize(args_fp, study_name="optimization", num_trials=20)

# # Run inference
# text = "Hello we will talk about the impact of transformer on chat GPT"
# run_id = open(Path(config.CONFIG_DIR, "run_id.txt")).read()
# predict_tag(text, run_id)
