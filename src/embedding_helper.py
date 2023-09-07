"""Helper class for getting embeddings from a model
"""
import os
from typing import List, Generator
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import torch
from torch import Tensor
import numpy as np
from src.logger import logger

os.environ["TOKENIZERS_PARALLELISM"] = "1"

class EmbeddingHelper:

    def __init__(self, model_id:str) -> None:
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModel.from_pretrained(self.model_id)
        logger.info("Loaded model %s", self.model_id)

    def average_pool(self, last_hidden_states:Tensor, attention_mask:Tensor) -> Tensor:
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
        return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


    def get_chunks(self, lst:List, chunk_size:int)->Generator[List, None, None]:
        for i in range(0, len(lst), chunk_size):
            yield lst[i:i+chunk_size]

    def get_embeddings_batch(self, input_texts:List, chunk_size:int=20)->List:
        logger.info(f'Getting embeddings for {len(input_texts)} texts')
        res = []
        for input_text in self.get_chunks(input_texts, chunk_size):
            batch_dict = self.tokenizer(input_text, max_length=512, padding=True, truncation=True, return_tensors='pt')
            outputs = self.model(**batch_dict)
            embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            embeddings = F.normalize(embeddings, p=2, dim=1)
            res.append(embeddings)

        embeddings = torch.cat(res, dim=0)
        embeddings_list = embeddings.tolist()
        return embeddings_list

    def get_embeddings_query(self, query: str, model_id: str)->Tensor:
        logger.info(f'Getting embeddings for {query}')
        batch_dict = self.tokenizer(query, max_length=512, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            outputs = self.model(**batch_dict)
        embeddings = self.average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings


    def manual_l2_distance(self, e1: np.array or list, e2: np.array or list)->float:
        '''
        Computes the L2 distance between two embeddings
        this is for testing purposes 
        '''
        e1 = np.array(e1)
        e2 = np.array(e2)
        l2 = np.sqrt(np.sum((e1-e2)**2))
        return 1 / (1 + l2**2)

