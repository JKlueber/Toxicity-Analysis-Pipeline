from ray_elasticsearch import ElasticsearchDatasource
from dotenv import load_dotenv
import os
from pyarrow import schema, field, string, bool_
from pathlib import Path

load_dotenv()

ELASTIC_PASSWORD = os.getenv('ELASTICSEARCH_PASSWORD')

def read_instances_from_file(config):
    file_path = Path(config['elasticsearch']['instances_file'])
    with open(file_path, 'r') as file:
        instances = [line.strip() for line in file if line.strip()]
    return instances

def get_es_source(config):
    instances = read_instances_from_file(config)

    return ElasticsearchDatasource(
        index=config['elasticsearch']['index'],
        hosts=config['elasticsearch']['host'],
        http_auth=(
            config['elasticsearch']['user'], 
            ELASTIC_PASSWORD,
        ),
        # verify_certs=False,
        timeout=60*60*24, # 1 day
        retry_on_timeout=True,
        max_retries=10,
        # connection_pool={"maxsize": 1000},
        keep_alive="24h",
        query={
            "bool": {
                "filter": [
                    {
                        "range": {
                            "crawled_at": {
                                "gte": config['date_range']['after'],
                                "lte": config['date_range']['before'],
                                "format": "date_hour_minute_second"
                            }                                                                                                                            
                        }
                    },
                    {
                        "terms": {
                            "crawled_from_instance": instances
                        }
                    },
                    {
                        "term": {
                            "language": config['elasticsearch']['language']
                        }
                    }
                ]
            }
        },
        schema=schema([
            field('_id', string()),
            field("content", string()),
            field("crawled_from_instance", string()),
            field("instance", string()),
            field("is_local", bool_()),
            field("created_at", string()),
            field("sensitive", bool_()),
            field("spoiler_text", string()),
        ]),
    )

def get_es_source_deduplication(config):
    instances = read_instances_from_file(config)
        
    return ElasticsearchDatasource(
        index=config['elasticsearch']['index'],
        hosts=config['elasticsearch']['host'],
        http_auth=(
            config['elasticsearch']['user'], 
            ELASTIC_PASSWORD,
        ),
        # verify_certs=False,
        timeout=60*60*24, # 1 day
        retry_on_timeout=True,
        max_retries=20,
        # connection_pool={"maxsize": 1000},
        keep_alive="24h",
        query={
            "bool": {
                "filter": [
                    {
                        "range": {
                            "created_at": {
                                "gte": config['date_range']['after'],
                                "lte": config['date_range']['before'],
                                "format": "date_hour_minute_second"
                            }                                                                                                                            
                        }
                    },
                    {
                        "terms": {
                            "crawled_from_instance": instances
                        }
                    },
                    {
                        "term": {
                            "language": config['elasticsearch']['language']
                        }
                    }
                ]
            }
        },
        schema=schema([
            field("content", string()),
            field("_id", string()),
        ]),
    )