DATASET = "evidence-inference"
DOCUMENT_COUNT_TO_PROCESS = 1000
K_VALUES = [1, 3, 5, 10]
LOG_DETAILED_RESULTS = False
SEARCH_ALL_CHUNKS = True

MODELS_DELOY = ["BAAI/bge-large-en-v1.5", "Snowflake/snowflake-arctic-embed-l-v2.0"]
SAGEMAKER_PARALLEL_WORKERS = 8