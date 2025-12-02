from azure.ai.ml import load_component
retrieval = load_component('components/feature_retrieval.yml')
print(retrieval)
