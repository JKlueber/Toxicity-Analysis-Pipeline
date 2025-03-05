from pathlib import Path
from datasketch import MinHash
import pandas as pd
import numpy as np
import pickle
import base64
import logging
import dask.dataframe as dd
import os

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

from ray.data import read_json
import ray

def load_data(data_path: str, pattern: str, concurrency: int = 100, num_blocks: int = 500) -> "ray.data.Dataset":
    file_paths = [str(file) for file in Path(data_path).glob(pattern)]
    
    ds = read_json(
        paths=file_paths,
        concurrency=concurrency,
        override_num_blocks=num_blocks,
        ray_remote_args=dict(num_cpus=0.01,
                             memory=10 * 1024**3)
    )
    return ds

# def write_data(df, output_path: str, concurrency: int = 100):
#     df_chunks = np.array_split(df, max(1, len(df) // 100_000))
#     ray_refs = [ray.put(chunk) for chunk in df_chunks]
#     ds_list = [ray.data.from_pandas(ray.get(ref)) for ref in ray_refs]
#     ds = ds_list[0]
#     for ds_chunk in ds_list[1:]:
#         ds = ds.union(ds_chunk)

#     ds.write_json(
#         path=output_path,
#         concurrency=concurrency,
#         ray_remote_args=dict(num_cpus=0.01)
#     )

def write_data(df, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "merged_results_2.parquet")
    df.to_parquet(output_path)
    logging.info(f"Data written to {output_path}")

def merge_data(df1, df2, on: str) -> pd.DataFrame:
    df1_dd = dd.from_pandas(df1, npartitions=min(500, max(100, len(df1) // 10000)))
    df2_dd = dd.from_pandas(df2, npartitions=min(500, max(100, len(df2) // 10000)))
    
    merged_df = df1_dd.merge(df2_dd, on=on, how="inner")
    
    return merged_df.compute()

def compute_hash(plaintext):
    m = MinHash()
    for word in plaintext.split():
        m.update(word.encode('utf8'))
    pickled = pickle.dumps(m)
    return base64.b64encode(pickled).decode('ascii')

def get_hash(batch: pd.DataFrame) -> pd.DataFrame:
    batch["hash"] = batch['plaintext'].apply(compute_hash)
    return batch

def insert_to_lsh(df: pd.DataFrame, lsh):
    minhashes = []
    for idx in range(len(df)):
        stored_hash_pickle = df.loc[idx, 'hash']
        pickled = base64.b64decode(stored_hash_pickle.encode('ascii'))
        m = pickle.loads(pickled)
        lsh.insert(f"doc_{idx}", m)
        minhashes.append(m)
    return minhashes


# def find_similar(minhashes, lsh):
#     to_remove = set()
#     for i, m in enumerate(minhashes):
#         similar_docs = lsh.query(m)
#         for doc_id in similar_docs:
#             doc_index = int(doc_id.split("_")[1])
#             if doc_index != i:
#                 to_remove.add(doc_index)
#     return to_remove

def find_similar(minhashes, lsh):
    similarity_map = {}
    group_id = 0
    
    for i, m in enumerate(minhashes):
        if i in similarity_map:
            continue 
        
        similar_docs = lsh.query(m)
        for doc_id in similar_docs:
            doc_index = int(doc_id.split("_")[1])
            if doc_index not in similarity_map:
                similarity_map[doc_index] = group_id
        
        similarity_map.setdefault(i, group_id)
        group_id += 1
    
    return similarity_map

def find_groups(df: pd.DataFrame, lsh) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    minhashes = insert_to_lsh(df, lsh)
    similarity_map = find_similar(minhashes, lsh)
    df["group"] = df.index.map(similarity_map)
    return df
