import gc
import pandas as pd
from datasketch import MinHashLSH, MinHash
import pickle

from functools import cached_property
import logging

from src.data_processing.dataset_loader import load_dataset

logging.basicConfig(level=logging.INFO)

class LSHBuilder:

    def _lsh_instance(self) -> MinHashLSH:
        return MinHashLSH(threshold=0.9, num_perm=128)

    def insert_to_lsh(self, batch: pd.DataFrame, lsh):
        for _, row in batch.iterrows():
            m = self.compute_hash(row['plaintext'])
            lsh.insert(row['_id'], m)
        return lsh
    
    def compute_hash(self, plaintext):
        m = MinHash(num_perm=128, seed=1)
        m.update_batch([s.encode('utf-8') for s in plaintext.split()])
        return m

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        lsh_empty = self._lsh_instance()
        lsh = self.insert_to_lsh(batch, lsh_empty)
        
        lsh_df = pd.DataFrame([{"lsh":lsh}])

        gc.collect()
        return lsh_df
    
class HashFinder:

    def __init__(self):
        self._lsh_instance = self._load_lsh_instance

    @cached_property
    def _load_lsh_instance(self) -> MinHashLSH:
        with open("/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-klueber/deduplicated_data/lsh_january.pkl", "rb") as file:
            lsh = pickle.load(file)
        return lsh
    
    def compute_hash(self, plaintext):
        m = MinHash(num_perm=128, seed=1)
        m.update_batch([s.encode('utf-8') for s in plaintext.split()])
        return m
    
    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        output_batch = []
        for _, row in batch.iterrows():
            hash = self.compute_hash(row['plaintext'])
            results = self._lsh_instance.query(hash)
            if row['_id'] == min(results):
                output_batch.append(row)

        gc.collect()
        return pd.DataFrame(output_batch, columns=batch.columns)
    
class MergeHash:

    def __init__(self):
        self._lsh_instance = self._load_lsh_instance
        self._data_to_merge = self._load_data_to_merge

    @cached_property
    def _load_data_to_merge(self) -> pd.DataFrame:
        data_path = "/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-klueber/toxicity_data/"
        pattern = "980_*.parquet"
        data_toxicity = load_dataset(data_path, pattern)
        data = data_toxicity.to_pandas()
        return data

    @cached_property
    def _load_lsh_instance(self) -> MinHashLSH:
        with open("/mnt/ceph/storage/data-in-progress/data-teaching/theses/thesis-klueber/deduplicated_data/lsh_january.pkl", "rb") as file:
            lsh = pickle.load(file)
        return lsh
    
    def compute_hash(self, plaintext):
        m = MinHash(num_perm=128, seed=1)
        m.update_batch([s.encode('utf-8') for s in plaintext.split()])
        return m
    
    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        tmp_batch = []
        for _, row in batch.iterrows():
            hash_value = self.compute_hash(row['plaintext'])
            results = self._lsh_instance.query(hash_value)
            if results:
                id = min(results) 
                merge_line = self._data_to_merge[self._data_to_merge['_id'] == id].drop(['_id', 'plaintext', 'language'], axis=1).squeeze()
                if not merge_line.empty:
                    output_line = pd.concat([row, merge_line], axis=0)
                    tmp_batch.append(output_line)

        columns=batch.columns.append(self._data_to_merge.columns).drop(['language']).drop_duplicates()
        output_batch = pd.DataFrame(tmp_batch, columns=columns)

        gc.collect()

        return output_batch