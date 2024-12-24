#!/usr/bin/env python
"""
Download from W&B the raw dataset and apply basic data cleaning, exporting the result to a new artifact
"""
import argparse
import logging
import os

import pandas as pd
import wandb

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):

    run = wandb.init(job_type="basic_cleaning")
    run.config.update(args)

    # Download input artifact. This will also log that this script is using this
    # particular version of the artifact
    # artifact_local_path = run.use_artifact(args.input_artifact).file()
    # Read artifact
    logger.info(f"Downloading  {args.input_artifact} from Weights&Biases to temp dir")
    artifact_path = run.use_artifact(args.input_artifact).file()
    df= pd.read_csv(artifact_path)
    
    # Drop duplicates
    logger.info("Dropping duplicates")
    df = df.drop_duplicates().reset_index(drop=True)
    # Drop outliers 
    logger.info(f'Drop outliers regarding min {args.min_price}, max {args.max_price} price thresholds')
    min_price = args.min_price
    max_price = args.max_price
    idx = df['price'].between(min_price, max_price)
    df = df[idx].copy()

    # Convert 'last_review' to datetime
    logger.info('Convert feature "last_review" to datetime type')
    df['last_review'] = pd.to_datetime(df['last_review'])
    
    # Drop rows in the dataset that are not in the proper geolocation
    logger.info('Drop rows in the dataset that are not in the proper geolocation')
    idx = df['longitude'].between(-74.25, -73.50) & df['latitude'].between(40.5, 41.2)
    df= df[idx].copy()
    
    # Save cleaned dataframe
    logger.info(f'Save cleaned dataframe to {args.output_artifact_name}')
    df.to_csv(args.output_artifact_name, index=False)

    # log artifact to Weights & Biases
    logger.info(f'W&B logging artifact {args.output_artifact_name}')
    
    artifact = wandb.Artifact(
        name=args.output_artifact_name,
        type=args.output_artifact_type,
        description=args.output_artifact_description,
    )
    artifact.add_file(args.output_artifact_name)
    run.log_artifact(artifact)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="This steps cleans the data")


    parser.add_argument(
        "--input_artifact", 
        type=str,
        help='Input artifact as a csv file)',
        required=True
    )

    parser.add_argument(
        "--output_artifact_name", 
        type=str,
        help='Output file name ',
        required=True
    )
    parser.add_argument(
        '--output_artifact_type', 
        type=str,
        help='Type of the output file',
        required=True
    )
    parser.add_argument(
        "--output_artifact_description",
        type=str,
        help="Description of the output artifact",
        required=True
    )
    parser.add_argument(
        '--min_price', 
        type=float,
        help='Minimum price to filter the price data for (e.g 10)',
        required=True
    )

    parser.add_argument(
        '--max_price', 
        type=float,
        help='Maximum price to filter the price data for (e.g 350)',
        required=True
    )
    args = parser.parse_args()

    go(args)
