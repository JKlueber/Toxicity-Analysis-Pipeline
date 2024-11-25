from ray_elasticsearch import ElasticsearchDatasource
from dotenv import load_dotenv
import os

load_dotenv()

elastic_password = os.getenv('ELASTICSEARCH_PASSWORD')

def get_es_source(config):
        
    return ElasticsearchDatasource(
        index=config['elasticsearch']['index'],
        client_kwargs=dict(
            hosts=config['elasticsearch']['host'],
            http_auth=(
                config['elasticsearch']['user'], 
                elastic_password,
            ),
            timeout=120
        ),
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
                    },
                    {
                    "term": {
                        "instance": "pawoo.net"
                    }
                    }
                ]
            }
        },
    )