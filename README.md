pipel# Build an ML Pipeline for Short-Term Rental Prices in NYC

This project builds a machine learning model to predict the typical price of a given short-term rental property based on the prices of similar properties. The company receives new data in bulk every week, and the model needs to be retrained on a weekly basis. Therefore, an end-to-end pipeline has been developed to automate this process, ensuring it can be reused for future data updates.

## Prerequisites

- Python (recommended version: 3.x)
- Jupyter Notebook (for interactive development)
- Linux environment (may be required on Windows, use WSL if needed)

## Pipeline Structure

![pipeline](https://github.com/user-attachments/assets/9bb322fc-3a4d-4c1c-9ff5-edafc7e637ce)

## Dependencies

This project uses the `mlflow` library, which handles package management and environment isolation for the pipeline. All required dependencies will be installed automatically within the environment managed by `mlflow`.

## Installation

To install `mlflow`, use the following command:

```bash
pip install mlflow
```
## Run Pipeline 
```bash
mlflow run https://github.com/lurui-star/build-ml-pipeline-for-short-term-rental-prices.git \
  -v 1.0.0 \
  -P hydra_options="etl.sample='sample2.csv'"
```
## Run Component of Pipeline 
```bash
mlflow run . -P hydra_options="main.steps=train_random_forest"
```
## License

Distributed under the MIT License. See [LICENSE](./LICENSE) for more information.
