import logging
from pathlib import Path

import typer
from animal_shelter.model import (
    split_x_y,
    fit_model,
    save_model,
    load_model,
    predict_model,
    save_results,
)
from animal_shelter.data import load_data

app = typer.Typer()


@app.callback()
def main() -> None:
    """Determine animal shelter outcomes."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)-15s] %(name)s - %(levelname)s - %(message)s",
    )


@app.command()
def train(input_path: Path, model_path: Path) -> None:
    """Trains a model on the given dataset."""

    logger = logging.getLogger(__name__)

    logger.info("Loading input dataset from %s", input_path)
    train_dataset = load_data(input_path)
    logger.info("Found %i rows", len(train_dataset))

    x_train, x_val, y_train, y_val = split_x_y(train_dataset)
    best_model, best_score = fit_model(x_train, y_train)
    logger.info("Final score %f", best_score)
    save_model(best_model, model_path)

    logger.info(f"Wrote model to {model_path}")


@app.command()
def predict(data_path: Path, output_path: Path, model_path: Path = 'output/trained_model.pkl') -> None:
    logger = logging.getLogger(__name__)

    outcome_model = load_model(model_path)
    logger.info(f"Loaded model to {model_path}")

    test_data = load_data(data_path)

    predictions = predict_model(outcome_model, test_data)

    save_results(predictions, output_path)
    logger.info(f"Wrote results to {output_path}")
