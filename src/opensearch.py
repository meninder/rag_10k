from typing import List
from opensearchpy import OpenSearch, NotFoundError
from tqdm import tqdm
from src.logger import logger


class OpenSearchClient:

    def __init__(self, model_id:str):
        self.client = self.get_client()
        self.body_indexer = self.get_body_indexer()
        self.index_name = self.get_index_name(model_id)
        logger.info(f'Using index name {self.index_name}')

    def get_index_name(self, model_id)->str:
        return model_id.replace('/','-') + '-index'
    
    def get_existing_indices(self)->List[str]:
        indices = self.client.cat.indices(format='json')
        indices = [i['index'] for i in indices]
        # self.client.indices.get_alias().keys() # alt way to get a list of indices
        #self.client.indices.get_mapping(index=model_id.replace('/', '-')+'-index')
        return indices

    def get_client(self,)->OpenSearch:
        client = OpenSearch(
            hosts = [{"host": "localhost", "port": 9200}],
            http_auth = ("admin", "admin"),
            use_ssl = True,
            verify_certs = False,
            ssl_assert_hostname = False,
            ssl_show_warn = False,
            )
        
        logger.info(client.info())

        return client

    def get_body_indexer(self)->dict:
        body = {
            "settings": {
                "index": {"knn": True},
            },
            "mappings": {
                "properties": {
                    "id": {"type": "integer"},
                    "filename":  {'type': 'keyword'},
                    #"content": {"type": "keyword"},
                    "embedding": {"type": "knn_vector", "dimension": 384, "method": {
                                  "name":"hnsw",
                                "engine":"lucene",
                                "space_type": "l2"}
                    },
                    "text": {"type": "text"},
                }
            },
        }   
        return body
    
    def get_body_query_concurrent(self, query_embedding:List, filename:str, k:int=2)->dict:
        '''
        Uses concurrent search, which doesn't return an intuitive score
        '''
        body = {
            "size": k,
            "query": {
                "bool":{
                    "must": [
                        {
                            "knn": {
                                "embedding": {
                                    "vector": query_embedding,
                                    "k": k
                                }
                            }
                        },
                        {
                            "match": {
                                "filename": filename
                            }
                        }
                    ]
                }
            }
        }
        return body
    
    def get_body_query_post_filter(self, query_embedding:List, filename:str, k:int=2)->dict:
        '''
        Uses post_filter, which returns intuitive scores, but might result in less hits
        '''
        body = {
            "size": k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_embedding,
                        "k": k,
                    }
                }
            },
            "post_filter": {"match": {"filename": filename}}
        }

        return body
    
    def get_body_query_efficient(self, query_embedding:List, filename:str, k:int=2)->dict:
        '''
        https://opensearch.org/docs/latest/search-plugins/knn/filter-search-knn/#efficient-k-nn-filtering
        '''
        body = {
            "size": k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_embedding,
                        "k": k,
                        "filter": {
                            "bool": {
                                "must": {"match": {"filename": filename}}
                                }
                            }
                        },
                }
            },
        }

        return body


    def create_index(self, overwrite:bool=False)->None:
        exists = self.index_name in self.get_existing_indices()
        if exists:
            logger.info(f'{self.index_name} exists')
            if overwrite:
                self.client.indices.delete(index=self.index_name)
                logger.info(f'deleted {self.index_name} because it already existed')
            else:
                logger.info(f'{self.index_name} exists and no overwrite flag was set, so not creating')
                return
        
        logger.info(f'Now creating {self.index_name}')    
        self.client.indices.create(self.index_name, body=self.body_indexer)


    def index_documents(self, embeddings_list:List, sub_documents:List[str], filename:str, chunk_size:int=10)->int:
        # TODO: add chunking
        for embeddings, sub_document in tqdm(zip(embeddings_list, sub_documents), total=len(embeddings_list)):
            self.client.index(
                index=self.index_name,
                body={
                    'filename': filename,
                    "embedding": embeddings,
                    "text": sub_document,
                    },
                )
        self.client.indices.refresh(index=self.index_name)
        logger.info("Done")
        logger.info(self.client.cat.count(index=self.index_name, format="json"))

    def search_documents(self, query_embedding, filename, k)->List:
        body = self.get_body_query_efficient(query_embedding, filename, k)
        res = self.client.search(index=self.index_name, body=body)
        return res
