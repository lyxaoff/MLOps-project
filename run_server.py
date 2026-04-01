import uvicorn
import argparse
from custom_property_api import create_app

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # This allows you to choose WHICH model version to put into production
    parser.add_argument("--model", type=str, default="property_evaluator_v1", 
                        help="The name of the model file to use for the API")
    
    args = parser.parse_args()
    
    # Create the app using the specific model version
    app = create_app(args.model)
    
    print(f"Launching Property API using model: {args.model}")
    uvicorn.run(app, host="127.0.0.1", port=8000)