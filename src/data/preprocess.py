import numpy as np
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from src.config import MODEL_DIR
import joblib


def split(df):
    df = df.copy()
    bad_idx = [523, 1298,1379]
    df = df.drop(index=bad_idx)
    df, df_holdout = train_test_split(df, test_size=0.15, random_state=42)
    return df,df_holdout

ordinal_cols = {
    "LotShape":        ["IR3", "IR2", "IR1", "Reg"],
    "LandSlope":       ["Sev", "Mod", "Gtl"],
    "LandContour":     ["Low", "HLS", "Bnk", "Lvl"],

    "ExterQual":       ["Po", "Fa", "TA", "Gd", "Ex"],
    "ExterCond":       ["Po", "Fa", "TA", "Gd", "Ex"],

    "BsmtQual":        ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtCond":        ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
    "BsmtExposure":    ["NA", "No", "Mn", "Av", "Gd"],
    "BsmtFinType1":    ["NA", "Unf", "LwQ", "BLQ", "Rec", "ALQ", "GLQ"],
    "BsmtFinType2":    ["NA", "Unf", "LwQ", "BLQ", "Rec", "ALQ", "GLQ"],

    "HeatingQC":       ["Po", "Fa", "TA", "Gd", "Ex"],
    "KitchenQual":     ["Po", "Fa", "TA", "Gd", "Ex"],

    "Functional":      ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],

    "FireplaceQu":     ["NA", "Po", "Fa", "TA", "Gd", "Ex"],

    "GarageFinish":    ["NA", "Unf", "RFn", "Fin"],
    "GarageQual":      ["NA", "Po", "Fa", "TA", "Gd", "Ex"],
    "GarageCond":      ["NA", "Po", "Fa", "TA", "Gd", "Ex"],

    "PavedDrive":      ["N", "P", "Y"],
    "Fence":           ["NA", "MnWw", "GdWo", "MnPrv", "GdPrv"],
}

nominal_cols = [
    "MSZoning","Alley","LotConfig","Neighborhood","Condition1",
    "BldgType","HouseStyle","RoofStyle","Exterior1st","Exterior2nd",
    "MasVnrType","Foundation","CentralAir","Electrical",
    "GarageType","SaleType","SaleCondition","MSSubClass"
]


def changes_df(df,df_holdout):
    df["MasVnrArea"] = df["MasVnrArea"].fillna(0)
    amenity_cols = [
        "PoolQC",        
        "MiscFeature",   
        "Alley",        
        "Fence",         
        "FireplaceQu",  
        "MasVnrType",   
        "GarageType",
        "GarageFinish",
        "GarageQual",
        "GarageCond",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2"
    ]
    df[amenity_cols] = df[amenity_cols].fillna("NA")
    df["GarageYrBlt"] = df["GarageYrBlt"].fillna(df["YearBuilt"])
    neighborhood_medians = df.groupby("Neighborhood")["LotFrontage"].median()
    df["LotFrontage"] = df["LotFrontage"].fillna(df["Neighborhood"].map(neighborhood_medians))
    df["MSSubClass"] = df["MSSubClass"].astype(str)
    df["SalePrice"] = np.log1p(df["SalePrice"])
    df_holdout["LotFrontage"] = df_holdout["LotFrontage"].fillna(df_holdout["Neighborhood"].map(neighborhood_medians))
    
    return df,df_holdout

def changes_df_holdout(df_holdout):
    df_holdout["MasVnrArea"] = df_holdout["MasVnrArea"].fillna(0)
    amenity_cols = [
        "PoolQC",        
        "MiscFeature",   
        "Alley",        
        "Fence",         
        "FireplaceQu",  
        "MasVnrType",   
        "GarageType",
        "GarageFinish",
        "GarageQual",
        "GarageCond",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2"
    ]
    df_holdout[amenity_cols] = df_holdout[amenity_cols].fillna("NA")
    df_holdout["GarageYrBlt"] = df_holdout["GarageYrBlt"].fillna(df_holdout["YearBuilt"])
    df_holdout["MSSubClass"] = df_holdout["MSSubClass"].astype(str)
    df_holdout["SalePrice"] = np.log1p(df_holdout["SalePrice"])

    return df_holdout

def pre_encoding(df,df_holdout):
    for col in ordinal_cols:
        df[col] = df[col].fillna("NA")
        df_holdout[col] = df_holdout[col].fillna("NA")

    return df,df_holdout


def encoding(df):
    ordinal_features = list(ordinal_cols.keys())
    ordinal_categories = [ordinal_cols[c] for c in ordinal_features]

    ord_enc = OrdinalEncoder(
        categories=ordinal_categories,
        handle_unknown="use_encoded_value",
        unknown_value=-1
    )

    ohe_enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("ord", ord_enc, ordinal_features),
            ("nom", ohe_enc, nominal_cols)
        ],
        remainder="passthrough"
    )

    pipe = Pipeline(
        steps=[("preprocess", preprocessor)]
    )

    path = MODEL_DIR / "encoding_pipe.pkl"
    pipe.fit(df.drop("SalePrice", axis=1))
    joblib.dump(pipe, path)
    return pipe