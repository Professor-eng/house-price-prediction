import numpy as np

def engineer_features(df):

    df = df.copy()
    df["StreetIsGrvl"] = (df["Street"] == "Grvl").astype(int)
    df["HeatingIsGasA"] = (df["Heating"] == "GasA").astype(int)
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]
    df['HouseRemodelAge'] = df['YrSold'] - df['YearRemodAdd']
    df["IsRemodeled"] = (df["YearRemodAdd"] != df["YearBuilt"]).astype(int)
    df["TotalLivingArea"] = df["GrLivArea"] + df["TotalBsmtSF"]
    df['TotalBaths'] = df['BsmtFullBath'] + df['FullBath'] + 0.5 * (df['BsmtHalfBath'] + df['HalfBath']) 
    df['TotalPorchSF'] = df['OpenPorchSF'] + df['3SsnPorch'] + df['EnclosedPorch'] + df['ScreenPorch'] + df['WoodDeckSF']
    df["HasBasement"] = (df["TotalBsmtSF"] > 0).astype(int)
    df["HasGarage"] = (df["GarageCars"] > 0).astype(int)
    df["FinishedBasementRatioPrimary"] = (df["BsmtFinSF1"] / df["TotalBsmtSF"].replace(0, np.nan)).fillna(0)
    df["FinishedBasementRatioTotal"] = ((df["BsmtFinSF1"] + df["BsmtFinSF2"]) / df["TotalBsmtSF"].replace(0, np.nan)).fillna(0)
    df["HasSecondFloor"] = (df["2ndFlrSF"] > 0).astype(int)
    df["MoSold_sin"] = np.sin(2 * np.pi * df["MoSold"] / 12)
    df["MoSold_cos"] = np.cos(2 * np.pi * df["MoSold"] / 12)
    porch_cols = [
        "WoodDeckSF",
        "OpenPorchSF",
        "EnclosedPorch",
        "3SsnPorch",
        "ScreenPorch"
    ]
    df["HasPorch"] = (df[porch_cols].sum(axis=1) > 0).astype(int)
    df["HasFireplace"] = (df["Fireplaces"] > 0).astype(int)
    df["GarageAge"] = df["YrSold"] - df["GarageYrBlt"]
    df["GarageAge"] = df["GarageAge"].clip(lower=0)
    df["PremiumRoof"] = df["RoofMatl"].isin(["WdShngl", "WdShake"]).astype(int)
    df["GasHeating"] = df["Heating"].isin(["GasA", "GasW"]).astype(int)
    
    
    cols_to_drop = ['Id','YrSold', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF', 'BsmtFinSF1', 'BsmtFinSF2', 'GrLivArea', 
                    'TotalBsmtSF','BsmtFullBath', 'FullBath', 'BsmtHalfBath', 'HalfBath', 'OpenPorchSF', '3SsnPorch', 'EnclosedPorch', 
                    'ScreenPorch','WoodDeckSF',"MoSold","MiscVal","MiscFeature","GarageArea","PoolArea","PoolQC","GarageYrBlt","Street","Utilities","RoofMatl",
                    "Condition2","Heating"]
    
    df = df.drop(columns=cols_to_drop)
    return df


