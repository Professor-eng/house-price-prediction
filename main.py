import pandas as pd
from src.data.make_dataset import load_raw_data, save_processed_data
from src.data.preprocess import split,changes_df,changes_df_holdout,pre_encoding
from src.features.build_features import engineer_features
import src.models.train as train
from src.models.evaluate import score_all
from src.models.stacking import train_stack
from src.data.preprocess import encoding
from src.config import MODEL_DIR
import joblib

def main():
    raw_data = load_raw_data()
    df, df_holdout = split(raw_data)
    df_processed, df_holdout = changes_df(df, df_holdout)
    df_holdout_processed = changes_df_holdout(df_holdout)
    df_processed = engineer_features(df_processed)
    df_holdout_processed = engineer_features(df_holdout_processed)
    df_processed, df_holdout_processed = pre_encoding(df_processed, df_holdout_processed)  # fixed
    encoding_pipe = encoding(df_processed)

    X, y = df_processed.drop("SalePrice", axis=1), df_processed["SalePrice"]
    X_holdout, y_holdout = df_holdout_processed.drop("SalePrice", axis=1), df_holdout_processed["SalePrice"]

    X_processed = pd.DataFrame(encoding_pipe.transform(X), columns=encoding_pipe.named_steps["preprocess"].get_feature_names_out())
    X_holdout_processed = pd.DataFrame(encoding_pipe.transform(X_holdout), columns=encoding_pipe.named_steps["preprocess"].get_feature_names_out())

    save_processed_data(X_processed, y.to_frame(), "train_X_fe.parquet", "train_y.parquet")
    save_processed_data(X_holdout_processed, y_holdout.to_frame(), "holdout_X.parquet", "holdout_y.parquet")

    models = train.train_all(X_processed, y)  
    for filename, model in models.items():
        train.save_model(model, f"{filename}_model.pkl")

    stack = train_stack(X_processed, y)        
    train.save_model(stack, "ensemble_model.pkl")
    models['stack'] = stack

    results = score_all(models, X_processed, y, X_holdout_processed, y_holdout)  
    print(results)

if __name__ == "__main__":
    main()