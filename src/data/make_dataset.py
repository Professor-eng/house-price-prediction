import pandas as pd
from src.config import DATA_RAW, DATA_PROCESSED

def load_raw_data(filename = "train_raw.csv"):
    return pd.read_csv(DATA_RAW/filename)

def save_processed_data(X_processed, y_processed , filename1, filename2):

    X_path = DATA_PROCESSED / filename1
    y_path = DATA_PROCESSED / filename2
    
    X_processed.to_parquet(X_path)
    if type(y_processed) == pd.Series:
        y_processed.to_frame().to_parquet(y_path) 
    else:
        y_processed.to_parquet(y_path)

def load_processed_data(filename1,filename2):
    X_path = DATA_PROCESSED / filename1
    y_path = DATA_PROCESSED / filename2

    return pd.read_parquet(X_path),pd.read_parquet(y_path)
