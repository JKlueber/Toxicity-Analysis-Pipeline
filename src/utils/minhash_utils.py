import gc
import pandas as pd
from pandas import DataFrame

from datasketch import MinHashLSH, MinHash
import pickle
from pathlib import Path

from ray.data import read_parquet_bulk
import ray

from functools import cached_property

from src.config.config_loader import load_config

# Computes MinHash for each batch
class CalculateMinHash:
    def compute_hash(self, plaintext):
        m = MinHash(num_perm=64, seed=1)
        m.update_batch([s.encode('utf-8') for s in plaintext.split()])
        return pickle.dumps(m) 

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        batch['minhash'] = batch['plaintext'].apply(self.compute_hash)

        return batch

# Creates LSH instances for each batch
class LSHBuilder:
    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        lsh = MinHashLSH(threshold=0.9, num_perm=64)
        with lsh.insertion_session() as session:
            for _, row in batch.iterrows():
                min_hash = pickle.loads(row['minhash'])
                session.insert(row['_id'], min_hash)
        
        lsh_df = pd.DataFrame([{"lsh":lsh}])

        gc.collect()
        return lsh_df

# Finds duplicates in the Dataset
class HashFinder:

    @cached_property
    def _lsh_instance(self) -> MinHashLSH:
        config = load_config()
        data_path = config["dir"]["lsh_dir"]
        with open(data_path+"lsh.pkl", "rb") as file:
            lsh = pickle.load(file)
        return lsh
    
    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        output_batch = []
        for _, row in batch.iterrows():
            min_hash = pickle.loads(row['minhash'])
            results = self._lsh_instance.query(min_hash)
            if row['_id'] == min(results):
                output_batch.append(row)

        gc.collect()
        return pd.DataFrame(output_batch, columns=batch.columns)

# Merges deduplicated Dataset with complete Dataset
class MergeHash:

    @cached_property
    def _data_to_merge(self) -> pd.DataFrame:
        config = load_config()
        data_path = config["dir"]["toxicity_analysis_dir"]
        file_paths = [str(file) for file in Path(data_path).glob("*.parquet")]
        
        ds = read_parquet_bulk(
            paths=file_paths,
            concurrency=10,
            columns=["_id", "toxicity", "severe_toxicity", "obscenity", "threat", "insult", "identity_attack", "sexually_explicit"],
            ray_remote_args=dict(
                num_cpus=0.01,
                memory=1 * 1024**3,
                max_retries=10,
                retry_exceptions=True,
            )
        )
        data = ds.to_pandas()
        return data

    @cached_property
    def _lsh_instance(self) -> MinHashLSH:
        config = load_config()
        data_path = config["dir"]["lsh_dir"]
        with open(data_path+"lsh.pkl", "rb") as file:
            lsh = pickle.load(file)
        return lsh
    
    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        tmp_batch = []

        for _, row in batch.iterrows():
            min_hash = pickle.loads(row['minhash'])
            results = self._lsh_instance.query(min_hash)
            if results:
                id = min(results) 
                merge_line = self._data_to_merge[self._data_to_merge['_id'] == id].drop(['_id'], axis=1).squeeze()
                if not merge_line.empty:
                    output_line = pd.concat([row, merge_line], axis=0)
                    tmp_batch.append(output_line)

        columns=batch.columns.append(self._data_to_merge.columns).drop_duplicates()
        output_batch = pd.DataFrame(tmp_batch, columns=columns)

        gc.collect()

        return output_batch

# Merges LSH instances on one index
@ray.remote(memory=50 * 1024**3)
class MergeLSHActor:
    def __init__(self):
        self.lsh = MinHashLSH(num_perm=64, threshold=0.9)

    def merge(self, batch: DataFrame) -> DataFrame:
        for instance in batch["lsh"]:
            self.lsh.merge(instance)

        gc.collect()
        
        output_batch = pd.DataFrame()
        return output_batch
    
    def store_lsh(self, output_path) -> MinHashLSH:
        with open(output_path, "wb") as file:
            pickle.dump(self.lsh, file)