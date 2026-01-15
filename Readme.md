### Dependencies
`pip install torch sentence-transformers nltk numpy psutil`.

### Main
`python main.py` will run the evaluation pipeline on `MODELS_TO_TEST` models configurations. It can be modified as needed.

After `main.py` is complete, results will be printed and also saved to `results/` folder as json files.

### config.py
1. `DATASET` variable determines which `data.json` evaluation runs on. `DATASET` must be a folder in `datasets/`.
2. `DOCUMENT_COUNT_TO_PROCESS` sets the limit to how many documents to process in the dataset
3. `DELAY_ON_EMBEDDING_CACHE_CLEAR` adds a delay and cache clear + garbage collection after each document is indexed. If not necessary on your machine, set to None or 0.
4. `LOG_DETAILED_RESULTS` will save retrieved chunks and queries to the json file after the run is complete.
5. `SEARCH_ALL_CHUNKS` - should lookup for relevant chunks to a given query happen across the entire library (if `True`) or only across chunks from its 's document (if `False`)



