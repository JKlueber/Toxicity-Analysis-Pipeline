from pathlib import Path

from ray_elasticsearch import ElasticsearchDatasource

def get_es_source(config):
    password_file = Path(config['elasticsearch']['password_file']).expanduser()
    with password_file.open("r") as f:
        password = f.readline().strip("\n")
        
    return ElasticsearchDatasource(
        index=config['elasticsearch']['index'],
        client_kwargs=dict(
            hosts=config['elasticsearch']['host'],
            http_auth=(
                config['elasticsearch']['user'], 
                password,
            ),
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