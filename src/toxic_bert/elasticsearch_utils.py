from ray_elasticsearch import ElasticsearchDatasource
from dotenv import load_dotenv
import os
from pyarrow import schema, field, string, bool_

load_dotenv()

ELASTIC_PASSWORD = os.getenv('ELASTICSEARCH_PASSWORD')

def get_es_source(config):
        
    return ElasticsearchDatasource(
        index=config['elasticsearch']['index'],
        hosts=config['elasticsearch']['host'],
        http_auth=(
            config['elasticsearch']['user'], 
            ELASTIC_PASSWORD,
        ),
        timeout=120,
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
        ]),
    )