from azure.ai.ml import MLClient, dsl, Input
from azure.identity import DefaultAzureCredential
from azure.ai.ml import load_component

# 1) CONNECT TO WORKSPACE
ml_client = MLClient(
    credential=DefaultAzureCredential(),
    subscription_id="a00dcbea-fd05-4973-82dc-120208b60116",
    resource_group_name="rg-60300294",    
    workspace_name="playeranalyticsml",   
)

# 2) LOAD COMPONENTS (A, B, C) FROM YAML
# Make sure these paths are correct relative to pipeline_job.py
feature_retrieval = load_component("components/feature_retrieval.yml")
feature_selection = load_component("components/feature_selection.yml")
train_eval = load_component("components/train_eval.yml")

# 3) DEFINE PIPELINE
@dsl.pipeline(
    default_compute="compute1",          
    experiment_name="player_pipeline",
)
def player_pipeline(gold_data: Input(type="uri_folder")):
    # Component A – Feature Retrieval
    retr = feature_retrieval(
        gold_data=gold_data
    )

    # Component B – Feature Selection
    fs = feature_selection(
        train_input=retr.outputs.train_output
    )

    # Component C – Training & Evaluation
    train_job = train_eval(
        train_input=retr.outputs.train_output,
        test_input=retr.outputs.test_output,
        selected_features=fs.outputs.selected_features_output,
    )

    # Expose useful outputs
    return {
        "metrics": train_job.outputs.metrics_output,
        "model": train_job.outputs.model_output,
    }


# 4) SUBMIT PIPELINE
if __name__ == "__main__":
    job = player_pipeline(
        gold_data=Input(
            type="uri_folder",
            path="azureml:playeranalytics_DataAssest:1",  # your GOLD data asset
        )
    )

    returned = ml_client.jobs.create_or_update(job)
    print("PIPELINE SUBMITTED ✔")
    print("Run name:", returned.name)
    print("View in Azure ML Studio:", returned.studio_url)
