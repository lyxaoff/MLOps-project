# MLOps Project
#**PRODUCT PRESENTATION AND OBJECTIVES**

In today's housing market, future landlords struggle to find fairly priced properties. For most buyers, the real estate market is quite vague. They don't know if a listing price is justified and quite accurate compared to the price of other similar properties. Moroever,they usually go through real estate agencies. Thus, it can take time, the suggested properties might not fit what they are looking for and their agencies might not be aware of all the properties available on the market.

Thus, we decided to create a product that would solve those issues: Property Evaluator API.

Our product has multiple objectives and advantages:
- It reduces information asymmetry by providing potential buyers with an estimation of the "Fair Value" of a property.
- We help buyers filter the properties according to the geographical areas they are targeting and their financial constraints to make them gain time in their research for realistic leads.
- We gain our customer's trust by eliminating their fear of overpaying and allowing users to make offers with the speed and certainty of a professional real estate agent.
- We are able to group all the properties available even if they are registered at different real estate agencies.

#**REPOSITORY STRUCTURE AND EXPLANATION**

This project is structured  as a MLOps pipeline so the Training Lifecycle and Production Runtime are separated.
Thus, we have the following files:
- model_trainer.py: automates data ingestion from the Ames Housing Dataset, executes Pycaret's preprocessing pipelines and exports the best-performing LightGBM model.
- property_evaluator_api.py: It's the FastAPI-powered service that defines the logic for loading the model and the modules in the FastAPI (prediction and filtering).
- run_server.py: dedicated entry point to launch the Uvicorn server allowing for environement-specific configurations (host, port and reloading).
- requirements.txt: defined environement file ensuring parity between development and production
- .gitignore: prevents temporary files, caches and large datasets from cluttering the version control system.
 README.md: presentation and instructions to use this product.


Concerning the utilization of the product, it is solely based on the Ames Housing Dataset for the moment so it concerns only people looking for a property in Ames. The user has the possibility to add preferences such as a max price and a neighborhood to obtain the "Fair Value" of properties within their criterion.

#**TECHNICAL ENVIRONMENT**

To participate in the improvement of the product or run the engine locally:

- Environment setup (install the necessary dependencies to ensure environment parity) : pip install -r requirements.txt
- Training the engine (retrain the model on the latest data): python model_trainer.py
- Launching the service(start FastAPI server and load the trained model): python run_server.py
- Saving a custom version with another model: python model_trainer.py --version property_evaluator_v2_lightgbm
- Deploying this custom version with another model: python run_server.py --model property_v2_lightgbm
- Open the API once launched: write http://127.0.0.1:8000/docs in your browser

#**NEXT STEPS**

In order to improve this product we can:
- Integrate AI to monitor the live data coming into your API and compare it to the training data to keep the "Fair Value" estimations accurate.
- To handle the high-traffic demands, move away from local execution by wrapping the files into Docker images and deploying containers via Kubernetes (K8s).
- Add more possibilities for filtering with new criterion.
- Instead of just predicting a price for one house, query a database (with SQL) to return the "Top 10 Best Matches" that are priced below or equal to the predicted Fair Value.
- Scale our product globally and not just for Ames neighborhood.
- Use Kubeflow or Airflow to trigger model_trainer.py automatically when new market data is ingested
- Implement Prometheus & Grafana dashboards to track API latency.