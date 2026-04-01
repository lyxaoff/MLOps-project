
import pandas as pd
from pycaret.regression import load_model, predict_model
from pycaret.datasets import get_data
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import numpy as np
from numpy import nan 

class input_model(BaseModel):
    Id: int = 61
    MSSubClass: int = 20
    MSZoning: str = 'RL'
    LotFrontage: Optional[float] = 63.0
    LotArea: int = 13072
    Street: str = 'Pave'
    Alley: Optional[str] = None
    LotShape: str = 'Reg'
    LandContour: str = 'Lvl'
    Utilities: str = 'AllPub'
    LotConfig: str = 'Inside'
    LandSlope: str = 'Gtl'
    Neighborhood: str = 'SawyerW'
    Condition1: str = 'RRAe'
    Condition2: str = 'Norm'
    BldgType: str = '1Fam'
    HouseStyle: str = '1Story'
    OverallQual: int = 6
    OverallCond: int = 5
    YearBuilt: int = 2004
    YearRemodAdd: int = 2004
    RoofStyle: str = 'Gable'
    RoofMatl: str = 'CompShg'
    Exterior1st: str = 'VinylSd'
    Exterior2nd: str = 'VinylSd'
    MasVnrType: Optional[str] = None
    MasVnrArea: float = 0.0
    ExterQual: str = 'TA'
    ExterCond: str = 'TA'
    Foundation: str = 'PConc'
    BsmtQual: str = 'Gd'
    BsmtCond: str = 'TA'
    BsmtExposure: str = 'No'
    BsmtFinType1: str = 'ALQ'
    BsmtFinSF1: int = 941
    BsmtFinType2: str = 'Unf'
    BsmtFinSF2: int = 0
    BsmtUnfSF: int = 217
    TotalBsmtSF: int = 1158
    Heating: str = 'GasA'
    HeatingQC: str = 'Ex'
    CentralAir: str = 'Y'
    Electrical: str = 'SBrkr'
    firstFlrSF: int = 1158
    secondFlrSF: int = 0
    LowQualFinSF: int = 0
    GrLivArea: int = 1158
    BsmtFullBath: int = 1
    BsmtHalfBath: int = 0
    FullBath: int = 1
    HalfBath: int = 1
    BedroomAbvGr: int = 3
    KitchenAbvGr: int = 1
    KitchenQual: str = 'Gd'
    TotRmsAbvGrd: int = 5
    Functional: str = 'Typ'
    Fireplaces: int = 0
    FireplaceQu: Optional[str] = None
    GarageType: str = 'Detchd'
    GarageYrBlt: Optional[float] = 2006.0
    GarageFinish: str = 'Unf'
    GarageCars: int = 2
    GarageArea: int = 576
    GarageQual: str = 'TA'
    GarageCond: str = 'TA'
    PavedDrive: str = 'Y'
    WoodDeckSF: int = 0
    OpenPorchSF: int = 50
    EnclosedPorch: int = 0
    threeSsnPorch: int = 0
    ScreenPorch: int = 0
    PoolArea: int = 0
    PoolQC: Optional[str] = None
    Fence: Optional[str] = None
    MiscFeature: Optional[str] = None
    MiscVal: int = 0
    MoSold: int = 5
    YrSold: int = 2006
    SaleType: str = 'New'
    SaleCondition: str = 'Partial'

class output_model(BaseModel):
    prediction: float

def create_app(model_name: str):
    app = FastAPI(title="Property Evaluator API")
    model = load_model(model_name)
    listing = get_data('house', verbose=False)

    @app.post("/predict", response_model=output_model)
    def predict(data: input_model):
        df = pd.DataFrame([data.model_dump()])
        df.columns = [c.replace('first', '1st').replace('second', '2nd').replace('three', '3Ssn') for c in df.columns]
        df.columns = [c.replace('3SsnSsn', '3Ssn') for c in df.columns]
        predictions = predict_model(model, data=df)
        return {"prediction": float(predictions["prediction_label"].iloc[0])}
    
    @app.get("/search-properties")
    def search_properties(max_price: float, neighborhood: str):
        local_listings = listing[listing['Neighborhood'] == neighborhood].copy()
        if local_listings.empty:
            return {"message": f"No houses found in neighborhood: {neighborhood}"} 
        predictions = predict_model(model, data=local_listings)
        matches = predictions[predictions['prediction_label'] <= max_price].copy()
        if matches.empty:
            return {"message": "No houses found within budget in this area."}
        matches = matches.sort_values(by='OverallQual', ascending=False)
        results = matches[['Id', 'Neighborhood', 'GrLivArea', 'OverallQual', 'prediction_label']]
        return results.to_dict(orient='records')
    return app
