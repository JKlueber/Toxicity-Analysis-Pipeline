from elasticsearch import Elasticsearch
from elasticsearch_dsl import Search, A
from elasticsearch_dsl.query import Range
from elasticsearch.helpers import scan
from pathlib import Path

def connect_to_elastic(config):
    password_file = Path(config['elasticsearch']['password_file']).expanduser()
    with password_file.open("r") as f:
        password = f.readline().strip("\n")
    es = Elasticsearch(
        [config['elasticsearch']['host']],
        port=config['elasticsearch']['port'],
        http_auth=(config['elasticsearch']['user'], password),
        timeout=3000,
        scheme="https"
    )
    return es

def prepare_search_query(es, config):
    date_query = Range(crawled_at={
        "gte": config['date_range']['after'],
        "lte": config['date_range']['before'],
        "format": "date_hour_minute_second"
    })
    base_search = Search(using=es, index=config['elasticsearch']['index']).filter(date_query)
    return base_search

def execute_scan(filtered_search, es, index, size=1000):
    return scan(client=es, query=filtered_search.to_dict(), index=index, size=size)

def get_all_instances(es, config):
    aggregation = A('terms', field='instance.keyword', size=1)
    search = Search(index=config['elasticsearch']['index']).using(es)
    search.aggs.bucket('unique_instances', aggregation)
    response = search.execute()
    
    instances = []
    buckets = response.aggregations.unique_instances.buckets
    for bucket in buckets:
        instances.append(bucket.key)
    
    return instances