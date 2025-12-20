"""
Multi-dataset loader supporting small, medium, and large-scale datasets
Includes automatic download, caching, and memory-efficient loading
"""

import os
import gzip
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import requests
import warnings

# Import config
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    DATASETS, RAW_DATA_DIR, PROCESSED_DATA_DIR, 
    RANDOM_STATE, CHUNK_SIZE, MEMORY_LIMIT_GB
)


def download_file(url, destination, desc="Downloading"):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    with open(destination, 'wb') as f, tqdm(
        desc=desc,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)
    
    print(f"✓ Downloaded to {destination}")


def download_kaggle_dataset(dataset_name, destination_dir):
    """
    Download dataset from Kaggle using kaggle API
    Requires: pip install kaggle and Kaggle API credentials setup
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        
        print(f"Downloading Kaggle dataset: {dataset_name}")
        api.dataset_download_files(
            dataset_name, 
            path=destination_dir, 
            unzip=True
        )
        print(f"✓ Downloaded Kaggle dataset to {destination_dir}")
        
    except ImportError:
        raise ImportError(
            "Kaggle package not installed. Please run: pip install kaggle\n"
            "Also setup Kaggle API credentials: https://www.kaggle.com/docs/api"
        )
    except Exception as e:
        raise Exception(
            f"Failed to download from Kaggle: {e}\n"
            f"Please download manually from: https://www.kaggle.com/datasets/{dataset_name}\n"
            f"And place the CSV file in: {destination_dir}"
        )


def load_breast_cancer_dataset(test_size=0.2, random_state=RANDOM_STATE):
    """Load breast cancer dataset (small - 569 samples)"""
    print("\n" + "="*60)
    print("Loading: Breast Cancer Wisconsin Dataset")
    print("="*60)
    
    data = load_breast_cancer()
    X = data.data
    y = data.target
    
    print(f"Samples: {X.shape[0]:,}")
    print(f"Features: {X.shape[1]}")
    print(f"Classes: {len(np.unique(y))} (balanced)")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Train samples: {X_train.shape[0]:,}")
    print(f"Test samples: {X_test.shape[0]:,}")
    print("="*60 + "\n")
    
    return X_train, X_test, y_train, y_test


def load_creditcard_fraud_dataset(test_size=0.2, random_state=RANDOM_STATE):
    """Load credit card fraud dataset (medium - ~284K samples)"""
    print("\n" + "="*60)
    print("Loading: Credit Card Fraud Detection Dataset")
    print("="*60)
    
    dataset_info = DATASETS["creditcard_fraud"]
    raw_file = RAW_DATA_DIR / dataset_info["filename"]
    
    # Check if dataset exists, if not download
    if not raw_file.exists():
        print("⚠ Dataset not found locally. Attempting download...")
        
        # Try Kaggle API download
        try:
            download_kaggle_dataset(
                dataset_info["kaggle_dataset"],
                RAW_DATA_DIR
            )
        except Exception as e:
            print(f"\n❌ Automatic download failed: {e}")
            print("\n📥 MANUAL DOWNLOAD REQUIRED:")
            print(f"1. Visit: {dataset_info['url']}")
            print(f"2. Download the dataset")
            print(f"3. Place 'creditcard.csv' in: {RAW_DATA_DIR}")
            raise FileNotFoundError(f"Please download dataset manually to: {raw_file}")
    
    # Load dataset
    print(f"Reading from: {raw_file}")
    df = pd.read_csv(raw_file)
    
    # Separate features and target
    target_col = dataset_info["target_column"]
    X = df.drop(columns=[target_col]).values
    y = df[target_col].values
    
    print(f"Samples: {X.shape[0]:,}")
    print(f"Features: {X.shape[1]}")
    print(f"Fraud cases: {y.sum():,} ({100*y.mean():.3f}%)")
    print(f"Class imbalance: {dataset_info['class_balance']}")
    
    # Stratified split due to high imbalance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Train samples: {X_train.shape[0]:,}")
    print(f"Test samples: {X_test.shape[0]:,}")
    print(f"Train fraud rate: {100*y_train.mean():.3f}%")
    print("="*60 + "\n")
    
    return X_train, X_test, y_train, y_test


def load_higgs_dataset(test_size=0.2, random_state=RANDOM_STATE, sample_size=None):
    """
    Load HIGGS boson dataset (large - 11M samples)
    
    Args:
        test_size: Fraction for test set
        random_state: Random seed
        sample_size: Optional - use subset for faster experiments (e.g., 1000000 for 1M samples)
    """
    print("\n" + "="*60)
    print("Loading: HIGGS Boson Dataset")
    print("="*60)
    
    dataset_info = DATASETS["higgs"]
    raw_file = RAW_DATA_DIR / dataset_info["filename"]
    
    # Check if dataset exists, if not download
    if not raw_file.exists():
        print("⚠ Dataset not found locally.")
        print(f"⚠ WARNING: This dataset is ~7.5GB and will take time to download!")
        
        response = input("Download now? (yes/no): ").lower().strip()
        if response != 'yes':
            print("❌ Download cancelled.")
            print(f"\n📥 To download manually:")
            print(f"1. Visit: {dataset_info['url']}")
            print(f"2. Save as: {raw_file}")
            raise FileNotFoundError(f"Dataset not found: {raw_file}")
        
        download_file(dataset_info["url"], raw_file, desc="Downloading HIGGS dataset")
    
    print(f"Reading from: {raw_file}")
    
    if sample_size:
        print(f"⚠ Using subset: {sample_size:,} samples (for faster experiments)")
        warnings.warn(
            f"Loading only {sample_size:,} samples instead of full {dataset_info['samples']:,} samples",
            UserWarning
        )
    
    # Memory-efficient loading for large dataset
    print("Loading data (this may take a few minutes)...")
    
    # Read compressed CSV
    with gzip.open(raw_file, 'rt') as f:
        if sample_size:
            # Load subset
            df = pd.read_csv(f, header=None, nrows=sample_size)
        else:
            # Load full dataset
            df = pd.read_csv(f, header=None)
    
    # First column is target, rest are features
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values.astype(int)
    
    print(f"Samples loaded: {X.shape[0]:,}")
    print(f"Features: {X.shape[1]}")
    print(f"Signal events: {y.sum():,} ({100*y.mean():.2f}%)")
    print(f"Dataset size in memory: ~{X.nbytes / (1024**3):.2f} GB")
    
    # Stratified split
    print("Splitting train/test...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"Train samples: {X_train.shape[0]:,}")
    print(f"Test samples: {X_test.shape[0]:,}")
    print("="*60 + "\n")
    
    return X_train, X_test, y_train, y_test


def load_dataset(dataset_name="breast_cancer", test_size=0.2, random_state=RANDOM_STATE, **kwargs):
    """
    Universal dataset loader
    
    Args:
        dataset_name: One of ['breast_cancer', 'creditcard_fraud', 'higgs']
        test_size: Fraction for test set (default 0.2)
        random_state: Random seed for reproducibility (default 42)
        **kwargs: Additional dataset-specific arguments (e.g., sample_size for HIGGS)
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    if dataset_name not in DATASETS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}\n"
            f"Available datasets: {list(DATASETS.keys())}"
        )
    
    # Display dataset info
    info = DATASETS[dataset_name]
    print(f"\n📊 Dataset: {info['name']}")
    print(f"   Source: {info['source']}")
    print(f"   Expected samples: {info['samples']:,}")
    print(f"   Features: {info['features']}")
    print(f"   Description: {info['description']}")
    
    # Route to appropriate loader
    if dataset_name == "breast_cancer":
        return load_breast_cancer_dataset(test_size, random_state)
    elif dataset_name == "creditcard_fraud":
        return load_creditcard_fraud_dataset(test_size, random_state)
    elif dataset_name == "higgs":
        return load_higgs_dataset(test_size, random_state, **kwargs)
    else:
        raise NotImplementedError(f"Loader for {dataset_name} not yet implemented")


# Backward compatibility - keep original function name
def load_data(test_size=0.2, random_state=RANDOM_STATE):
    """
    Original function for backward compatibility
    Loads default breast cancer dataset
    """
    return load_dataset("breast_cancer", test_size, random_state)


def get_dataset_info(dataset_name):
    """Get information about a dataset"""
    if dataset_name not in DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return DATASETS[dataset_name]


def list_available_datasets():
    """Print all available datasets"""
    print("\n" + "="*80)
    print("AVAILABLE DATASETS FOR SCALABILITY ANALYSIS")
    print("="*80)
    
    for name, info in DATASETS.items():
        print(f"\n📊 {name}")
        print(f"   Name: {info['name']}")
        print(f"   Samples: {info['samples']:,}")
        print(f"   Features: {info['features']}")
        print(f"   Description: {info['description']}")
        print(f"   Balance: {info['class_balance']}")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    # Demo usage
    list_available_datasets()
    
    # Test loading breast cancer (small dataset)
    print("\n🧪 Testing small dataset loader...")
    X_train, X_test, y_train, y_test = load_dataset("breast_cancer")
    print(f"✓ Successfully loaded breast_cancer dataset\n")
