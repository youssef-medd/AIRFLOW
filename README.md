# AirFlow Inside

AirFlow Inside is a hybrid driving-risk detection project that combines sensor telemetry with image-based vision analysis to classify driving behavior into safe or unsafe patterns.

## Overview

The project uses:
- sensor features such as pressure, vibration, wind speed, and aerodynamic asymmetry
- a lightweight machine learning classifier for sensor-only inference
- a vision model for image-based scene analysis
- a fused inference pipeline for combined predictions

## Key Files

- [airflow.py](airflow.py) — trains the sensor classifier and optionally runs a fused inference smoke test
- [ai_inference.py](ai_inference.py) — loads saved models and performs inference on a sensor row plus an image
- [ai_integration.py](ai_integration.py) — shared model definitions, feature lists, and fusion logic
- [cv_train.py](cv_train.py) — trains the vision model
- [sensor_gen.py](sensor_gen.py) — generates synthetic sensor samples
- [sensor_data.csv](sensor_data.csv) — example labeled sensor dataset

## Requirements

Install dependencies with:

```bash
python -m pip install -r requirements.txt
```

## Quick Start

### 1. Train the sensor model

```bash
python airflow.py --model_type logistic
```

This creates:
- `model.pkl`
- `scaler.pkl`

### 2. Train the vision model (optional)

```bash
python cv_train.py --data /path/to/image_dataset
```

This produces:
- `cv_model.pt`

### 3. Run fused inference

```bash
python ai_inference.py \
  --sensor_model model.pkl \
  --cnn_model cv_model.pt \
  --scaler scaler.pkl \
  --img /path/to/example.jpg
```

## Dataset Notes

The expected sensor feature set is defined in [config.py](config.py) and includes:
- `front_corr`
- `sideL_corr`
- `sideR_corr`
- `d_front`
- `d_sideL`
- `d_sideR`
- `asym`
- `turbulence`
- `internal_pressure`
- `vibration`
- `wind_speed`

## Project Structure

- `airflow.py` — end-to-end model training workflow
- `ai_inference.py` — inference entry point
- `ai_integration.py` — model integration and fusion implementation
- `cv_dataloader.py` — image dataset loader for CV training
- `cv_model.py` — CNN model definitions
- `cv_preprocessing.py` — preprocessing utilities
- `cv_train.py` — training script for the vision branch
- `sensor_gen.py` — sample data generation

## Notes

- The current training pipeline expects a labeled CSV with a `label` column.
- If a vision checkpoint is unavailable, the fused workflow will skip the fused test and prompt you to train the vision model first.
- The project is intended for experimentation and prototype evaluation rather than production deployment.


