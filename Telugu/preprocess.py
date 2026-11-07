"""Compute mel, pitch, energy and duration files from audio + MFA TextGrids.

Usage:  python preprocess.py config.yaml
"""
import argparse

from preprocessor.preprocessor import Preprocessor
from utils.config import load_config

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to config.yaml")
    args = parser.parse_args()

    preprocess_config, _, _ = load_config(args.config)
    Preprocessor(preprocess_config).build_from_path()
