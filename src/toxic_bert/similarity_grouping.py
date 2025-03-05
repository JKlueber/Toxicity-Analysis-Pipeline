import ray
import gc
import base64
import pickle
import pandas as pd
from datasketch import MinHashLSH


@ray.remote
class GroupIDCounter:
    def __init__(self):
        self.global_group_counter = 0

    def get_next_id(self):
        curr_id = self.global_group_counter
        self.global_group_counter += 1
        return curr_id


class GroupBuilder:
    def __init__(self, counter_actor):
        self.counter = counter_actor

    def _lsh_instance(self) -> MinHashLSH:
        return MinHashLSH(threshold=0.9, num_perm=128)

    def insert_to_lsh(self, batch: pd.DataFrame, lsh):
        minhashes = []
        for idx in range(len(batch)):
            stored_hash_pickle = batch.loc[idx, 'hash']
            pickled = base64.b64decode(stored_hash_pickle.encode('ascii'))
            m = pickle.loads(pickled)
            lsh.insert(f"doc_{idx}", m)
            minhashes.append(m)
        return minhashes

    def find_similar(self, minhashes, lsh):
        similarity_map = {}

        for i, m in enumerate(minhashes):
            if i in similarity_map:
                continue

            similar_docs = lsh.query(m)

            group_id = ray.get(self.counter.get_next_id.remote())

            for doc_id in similar_docs:
                doc_index = int(doc_id.split("_")[1])
                if doc_index not in similarity_map:
                    similarity_map[doc_index] = group_id

            similarity_map.setdefault(i, group_id)

        return similarity_map

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        batch = batch.reset_index(drop=True)
        lsh = self._lsh_instance()
        minhashes = self.insert_to_lsh(batch, lsh)
        similarity_map = self.find_similar(minhashes, lsh)
        batch["group"] = batch.index.map(similarity_map)

        gc.collect()
        return batch

