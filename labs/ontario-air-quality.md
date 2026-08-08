# Probabilistic Ontario Air-Quality Forecasting Research Project

## Goal

Build a rigorous forecasting system for Ontario air quality while learning statistical modelling, time-series ML, uncertainty estimation and experimental methodology.

Unlike a normal prediction application, this project should emphasize understanding why models work and quantifying uncertainty.

## Initial research problem

Begin with a clearly defined target such as forecasting PM2.5 or AQHI several hours into the future for selected Ontario monitoring stations.

Do not immediately use complex deep-learning models.

## Forecasting protocol

Define the prediction problem before modelling.

Example:

- Target: PM2.5 concentration at monitoring station s
- Horizons: 1h, 3h, 6h, 12h, 24h
- Available information: only variables observable at forecast issue time
- Point metrics: MAE and RMSE
- Probabilistic metrics: pinball loss, coverage probability, interval width
- Splitting: chronological train/validation/test

## Stage 1 — Data

Build a dataset using historical air-quality measurements and relevant variables such as:

- previous pollutant measurements
- temperature
- humidity
- wind
- precipitation
- time of day
- season

Study missing data, seasonality, temporal correlation and distribution shift.

Use time-based train/validation/test splits rather than random splitting.

## Stage 2 — Baselines

Establish simple baselines:

- persistence
- historical averages
- linear regression
- ridge regression

A complex model should only be considered successful if it meaningfully beats appropriate simple baselines.

## Stage 3 — Classical ML

Investigate methods such as:

- nonlinear regression
- tree-based models
- engineered temporal features

Compare performance systematically.

## Stage 4 — Probabilistic prediction

Move beyond point estimates.

Learn and experiment with:

- predictive distributions
- quantile regression
- confidence/prediction intervals
- calibration
- negative log likelihood
- coverage probability

The model should communicate not only its prediction but how uncertain that prediction is.

## Stage 5 — Deep learning

Only after strong baselines exist, investigate sequence-based neural models.

Compare them against simpler approaches.

Do not assume a neural model should perform better.

## Stage 6 — Research questions

Investigate questions such as:

- How far ahead can air quality be predicted reliably?
- Which variables contribute most at different horizons?
- How does wildfire-smoke season change model behaviour?
- Does adding weather data materially improve predictions?
- Which models provide the best calibrated uncertainty?
- How badly does performance degrade during unusual events?

## Stage 7 — Final artifact

Produce:

- reproducible data pipeline
- multiple model families
- rigorous temporal validation
- uncertainty evaluation
- ablation experiments
- technical report
- clear visualizations
- documented negative results

## AI role

Do not allow AI to choose modelling decisions silently.

Before recommending a method, explain its assumptions and mathematics.

Ask me which approach I think we should take before giving your recommendation.

Make me reason about leakage, metrics, uncertainty and experimental design.

Treat the project like a small research investigation rather than a Kaggle competition.
