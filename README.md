# Burned Area Mapping using Random Forest and Sentinel-2

This project implements an end-to-end pipeline for mapping burned areas using
Sentinel-2 Level-2A satellite imagery and supervised machine learning. The main
classifier is a Random Forest model, with optional support for XGBoost and a
soft-voting RF + XGBoost ensemble.

The pipeline supports training, model registration, daily inference, historical
reprocessing, raster postprocessing, temporal aggregation, and MLflow experiment
logging.

## Features

- Sentinel-2 L2A SAFE ingestion and band loading
- SCL-based masking for clouds, cloud shadows, cirrus, and water
- Burn-sensitive spectral indices including BAI, GEMI, MIRBI, NBR, NBR2, NDMI,
  NDVI, NDWI, and SAVI
- Supervised classification using Random Forest, XGBoost, or RF + XGBoost
  ensemble
- Class imbalance handling and optional hyperparameter tuning
- Threshold optimization during training
- User-configured or model-registered probability threshold for inference
- Model registry with metadata, metrics, feature list, and optimized threshold
- MLflow experiment logging
- Per-tile daily inference outputs
- Historical SAFE archive reprocessing
- Temporal consistency filtering
- Minimum mapping unit (MMU) filtering
- Auxiliary mask support for water, urban, forest, and agriculture layers
- Final tile-level outputs for burn mask, maximum burn probability, first burn
  day-of-year, and latest FCC visualization

## Project Structure

```text
burnt-area-mapping-using-sentinel2/
|
|-- configs/
|   |-- train.yaml                 # Training data path, features, model settings
|   `-- inference.yaml             # Inference, threshold, and postprocessing settings
|
|-- scripts/
|   |-- run_training.py            # Train, evaluate, register, and log model
|   |-- run_daily_inference.py     # Run inference for today's SAFE folders
|   |-- run_reprocess_all.py       # Reprocess all available SAFE scenes
|   |-- run_aggregation.py         # Aggregate per-date outputs into final products
|   `-- organise_safe_archive.py   # Move SAFE archives into tile/date folders
|
|-- src/
|   |-- ingestion/
|   |   `-- sentinel_loader.py     # Sentinel-2 L2A band loading and SCL masking
|   |-- features/
|   |   `-- spectral_indices.py    # Spectral index and feature stack generation
|   |-- training/
|   |   |-- train.py               # Model training entry point
|   |   |-- split.py               # Stratified train/test splitting
|   |   |-- hyperparameter.py      # RandomizedSearchCV for Random Forest
|   |   |-- class_weights.py       # Imbalance handling helpers
|   |   |-- calibration.py         # Optional classifier calibration
|   |   `-- cross_validation.py    # Cross-validation utilities
|   |-- inference/
|   |   `-- predict_tile.py        # Per-SAFE raster prediction
|   |-- postprocessing/
|   |   |-- aggregate_tile_outputs.py # Temporal aggregation and final rasters
|   |   |-- aux_mask.py            # Auxiliary non-burnable mask alignment
|   |   |-- mmu.py                 # Minimum mapping unit filtering
|   |   `-- temporal_filter.py     # Temporal consistency filter
|   |-- evaluation/
|   |   |-- metrics.py             # Accuracy, F1, ROC AUC, PA/UA metrics
|   |   |-- threshold_optimization.py
|   |   |-- feature_importance.py
|   |   `-- shap_explain.py
|   |-- registry/
|   |   |-- model_registry.py      # Local model and metadata registry
|   |   |-- best_model_selector.py # Select best registered model by metric
|   |   |-- mlflow_registry.py     # MLflow logging
|   |   `-- experiment_summary.py
|   |-- monitoring/
|   |   `-- drift.py               # Population stability index
|   `-- utils/
|       |-- config.py              # YAML config loader
|       `-- logger.py              # Logger setup
|
|-- data/
|   |-- training/                  # Training CSV files
|   |-- raw/                       # Sentinel-2 SAFE folders grouped by tile/date
|   |-- processed/                 # Per-date burn probability/mask/FCC outputs
|   |-- outputs/                   # Final aggregated tile outputs
|   `-- auxillary/                 # Optional water/urban/forest/agriculture masks
|
|-- models/
|   |-- registry/                  # Registered model .pkl and .json metadata
|   `-- metrics/                   # Training metrics JSON
|
|-- tests/
|   `-- test_pipeline_fixes.py     # Regression tests for key pipeline behavior
|
|-- Makefile
|-- requirements.txt
`-- README.md
```

## Setup Instructions

### Clone the Repository

```bash
git clone https://github.com/ypraveen-mss/burnt-area-mapping-using-sentinel2.git
cd burnt-area-mapping-using-sentinel2
```

### Install Requirements

Using `venv`:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Using Conda:

```bash
conda create -n ba-rf-pipeline python=3.11
conda activate ba-rf-pipeline
pip install -r requirements.txt
```

## Configuration

### Training Config

Edit `configs/train.yaml` to set:

- Training CSV path
- Label column
- Feature list
- Model output path
- Hyperparameter tuning flag
- XGBoost / ensemble options

Example:

```yaml
csv_path: data/training/odisha_training_bhoonidhi.csv
label_col: Class
model_output_path: models/latest_model.pkl
tune: false
use_ensemble: true
```

### Inference Config

Edit `configs/inference.yaml` to set:

- Raw SAFE data root
- Processed output root
- Final aggregated output root
- Model registry path
- Probability threshold behavior
- Postprocessing options

By default, inference uses the user-defined threshold:

```yaml
threshold_source: user
threshold: 0.5
```

To use the optimized threshold stored with the selected registered model:

```yaml
threshold_source: registered
threshold: 0.5
```

In this mode, `threshold` is only a fallback when the selected model metadata
does not contain `optimal_threshold`.

## Data Preparation

### Training Data

Place labeled CSV training data under:

```text
data/training/
```

The CSV must contain:

- A binary label column, configured as `label_col` in `configs/train.yaml`
- Required spectral bands or renamed equivalents:
  - `B02`, `B03`, `B04`, `B08`, `B11`, `B12`
  - or `BLUE`, `GREEN`, `RED`, `NIR`, `SWIR1`, `SWIR2`
- Any spectral indices listed in `configs/train.yaml`

### Sentinel-2 SAFE Data

Raw SAFE folders are expected under:

```text
data/raw/<tile>/<DD-Mon-YYYY>/<SAFE folder>
```

Example:

```text
data/raw/T44QME/01-Feb-2026/S2B_MSIL2A_..._T44QME_....SAFE
```

If SAFE folders are staged under `data/archive`, organize them with:

```bash
python scripts/organise_safe_archive.py
```

## Running the Pipeline

### Train the Model

```bash
python scripts/run_training.py
```

This will:

- Read the training CSV
- Train the selected classifier
- Evaluate the model
- Optimize the probability threshold
- Save metrics under `models/metrics/`
- Register the model and metadata under `models/registry/`
- Log the experiment to MLflow

### Run Daily Inference

```bash
python scripts/run_daily_inference.py
```

This scans `data/raw/<tile>/<today>/` for SAFE folders and writes per-date
outputs to:

```text
data/processed/<tile>/
```

### Reprocess All Available SAFE Scenes

```bash
python scripts/run_reprocess_all.py
```

This scans every tile/date folder under `data/raw/` and runs inference for all
available SAFE scenes.

### Aggregate Tile Outputs

```bash
python scripts/run_aggregation.py
```

This creates final per-tile products under:

```text
data/outputs/<tile>/
```

## Makefile Commands

```bash
make train
make daily-inference
make reprocess
make aggregate
make test
```

## Outputs

Per-date inference outputs:

```text
data/processed/<tile>/<tile>_<YYYYMMDD>_BurntProb.tif
data/processed/<tile>/<tile>_<YYYYMMDD>_BurntMask.tif
data/processed/<tile>/<tile>_<YYYYMMDD>_Uncertainty.tif
data/processed/<tile>/<tile>_<YYYYMMDD>_FCC.tif
```

Final aggregated outputs:

```text
data/outputs/<tile>/<tile>_FinalBurnMask.tif
data/outputs/<tile>/<tile>_MaxBurnProb.tif
data/outputs/<tile>/<tile>_FirstBurnDOY.tif
data/outputs/<tile>/<tile>_LatestFCC.tif
```

## Evaluation and Monitoring

The training pipeline reports:

- Confusion matrix
- Overall accuracy
- F1 score
- ROC AUC
- Producer's accuracy
- User's accuracy
- Burnt-class producer's and user's accuracy

Additional utilities include:

- Feature importance
- SHAP explainability
- Cross-validation
- Probability threshold optimization
- Population stability index for drift monitoring

## Tests

Run regression tests with:

```bash
python -m unittest discover -s tests
```

Some tests require optional geospatial or ML dependencies such as `rasterio` and
`scikit-learn`. Tests that cannot import those dependencies are skipped.

## Future Extensions

- Add Sentinel-1 SAR features for cloud-independent burned area detection
- Add deep learning models for temporal burn dynamics
- Add automated report generation and summary plots
- Add cloud-native execution on object storage and batch compute
- Add CI checks for formatting, tests, and reproducible model metadata
- Add support for region-level mosaicking across multiple Sentinel-2 tiles

## Notes

- Training labels are expected to represent manually selected burnt and unburnt
  pixels across relevant land-cover types.
- Balanced, representative training data is important for stable burned-area
  classification.
- The default inference threshold source is `user`, so the configured threshold
  is used unless `threshold_source` is changed to `registered`.
- Large raw imagery, processed rasters, model artifacts, MLflow runs, and local
  databases are ignored by `.gitignore` and should generally not be committed.

## Authors and Credits

Developed by: Your Name / Team / Organization

Remote sensing data source: Sentinel-2 MSI Level-2A

Modeling approach: Random Forest, XGBoost, and ensemble classifiers

## License

MIT License. You are free to reuse, modify, and extend this project.

## Contact

For support, open a GitHub issue or contact:

```text
your.email@example.com
```
