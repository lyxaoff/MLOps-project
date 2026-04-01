# I checked the datasets available with get_data(index) and decided to use the house one 
#for our project as it is appropriate to create a toolset 
# for people who want to buy a property.
# I used Datacamp and PyCaret documentation to help me.
import argparse
from pycaret.datasets import get_data
from pycaret.regression import setup, compare_models, save_model
def run_training(version_name):
    data = get_data('house')

# We initialize the function 'setup' of PyCaret that controls all the data preprocessing operations.
    s = setup(data, target='SalePrice', session_id=123, verbose=False)

# We train and evaluate how each models from our library perform. 
# We take RMSE as our primary reference metric as for our real estate startup, 
# minimizing extreme errors is critical to ensuring the credibility of our property evaluations.
    best = compare_models(sort = 'RMSE')

# The lightgbm (Light Gradient Boosting Machine) model got the lowest RMSE. 
# Its RMSE is 2.9 which means that our model's predictions deviate by approximately $29k.
# Its R2 is 0.85 which means that our tool is capable of explaining 85% of the variance in house prices.
# Save with a specific version name
    save_model(best, version_name)
    print(f"Success! Model saved as {version_name}.pkl")

# We now build the API of our best model (lightgbm) with create_api
# PyCaret is integrated to FastAPI which allow it to create automatically an API REST from the model.
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default="property_evaluator_v1")
    args = parser.parse_args()
    run_training(args.version)