
from typing import Dict
import torch
from torch import nn, Tensor
from torch.utils.data import Dataset
from sentence_transformers.util import cos_sim
from llama_index.embeddings.base import BaseEmbedding

import os
import json
import pdb


from src.logger import logger

class LinearModel(nn.Module):
    """Linear transformation, no bias.
    will output the same dimension as the original embedding model
    """

    def __init__(self, embed_model: BaseEmbedding) -> None:
        super(LinearModel, self).__init__()
        dim = len(embed_model.get_text_embedding("hello world")) # get embedding dimension
        self.in_features = dim
        self.out_features = dim
        
        self.linear = nn.Linear(self.in_features, self.out_features, bias=False)
        
        # sets the initial matrix as the identity matrix
        self.linear.weight.data.copy_(torch.eye(self.in_features, self.out_features))


    def forward(self, embed: Tensor) -> Tensor:
        """Forward pass (Wv)."""
        return self.linear(embed)

    def get_config_dict(self) -> Dict:
        return {
            "in_features": self.in_features,
            "out_features": self.out_features,
            "bias": self.bias,
        }

    def save(self, output_path: str) -> None:
        """Save model."""
        os.makedirs(output_path, exist_ok=True)
        with open(os.path.join(output_path, "config.json"), "w") as fOut:
            json.dump(self.get_config_dict(), fOut)
        torch.save(self.state_dict(), os.path.join(output_path, "pytorch_model.bin"))

    @staticmethod
    def load(input_path: str) -> "LinearModel":
        """Load model."""
        with open(os.path.join(input_path, "config.json")) as fIn:
            config = json.load(fIn)
        model = LinearModel(**config)
        model.load_state_dict(
            torch.load(
                os.path.join(input_path, "pytorch_model.bin"),
                map_location=torch.device("cpu"),
            )
        )
        return model


class LossFunction(nn.Module):
    """Multiple negatives ranking loss.  Used by sentence transformers.
    That implementation uses sentence text and a model to embed.  This implementation uses embeddings directly.
    https://www.sbert.net/docs/package_reference/losses.html#multiplenegativesrankingloss
    """

    def __init__(self, model, scale: float = 20.0, similarity_fct = None, device='cpu'):
        """Define ranking loss."""

        super(LossFunction, self).__init__()
        self.model = model
        self.scale = scale # sentence transformers uses 20.0 for cosine, and suggests 1.0 for dot product   
        self.similarity_fct = cos_sim if similarity_fct is None else similarity_fct
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.device = device

    def forward(self, query_embeds, context_embeds):
        """Forward pass."""
        context_embeds = context_embeds.to(self.device)
        query_embeds = query_embeds.to(self.device)


        # apply the adapter to the output
        query_embeds_2 = self.model.forward(query_embeds)  

        # compute similarity of new embeddings to original contexts.
        # creates a (batch_size x batch_size) tensor where each row and col is a sample.
        scores = self.similarity_fct(query_embeds_2, context_embeds) * self.scale

        logger.debug(f"Scores: {scores}")
        logger.debug(f"Scores shape: {scores.shape}") # batch_size x batch_size

        # The true label for each sample is its index 
        # eg the i_th col for the i_th sample should have the highest score
        labels = torch.tensor(
            range(len(scores)), dtype=torch.long, device=scores.device
        )
        logger.debug(f"Labels: {labels}")
        return self.cross_entropy_loss(scores, labels)

    def accuracy(self, dl, ignore_keys=False):
        logger.debug('**MSP: Computing accuracy inside loss function')
        # pdb.set_trace()
        
        num_correct = 0
        total = 0
        data_iterator = iter(dl)
        for query_embeds, context_embeds in data_iterator:
            context_embeds = context_embeds.to(self.device)
            query_embeds = query_embeds.to(self.device)
            with torch.no_grad():
                query_embeds_2 = self.model.forward(query_embeds)
                scores = self.similarity_fct(query_embeds_2, context_embeds)
                num_correct += sum([i.item()==j.item() for i,j in zip(scores.argmax(dim=1), torch.arange(len(scores)))])
                total += len(scores)
        return num_correct/total

    def topk(self, dl, ignore_keys=False):
        logger.debug('**MSP: Computing topk inside loss function')
        num_correct = 0
        total = 0
        data_iterator = iter(dl)
        for query_embeds, context_embeds in data_iterator:
            context_embeds = context_embeds.to(self.device)
            query_embeds = query_embeds.to(self.device)
            with torch.no_grad():
                query_embeds_2 = self.model.forward(query_embeds)
                scores = self.similarity_fct(query_embeds_2, context_embeds)
                top_k = torch.topk(scores, k=2, dim=1)
                labels = torch.arange(len(scores))
                assert top_k.indices.shape[0] == len(labels)
                for result, label in zip(top_k.indices, labels):
                    if label in result:
                        num_correct += 1
                total += len(scores)
        logger.debug(f'**MSP: Num topk: {num_correct}, total: {total}')
        return num_correct/total

    
def optimizer(model, lr: float = 1e-3):
    """Define optimizer."""
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()],
        },
    ]
    return torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)

def collate_fn(batch, embed_model):
    """
    Collate function that embeds the texts and create a tensor of embeddings
    """
    from torch import Tensor
    import torch

    query_embeddings = []
    text_embeddings = []
    logger.debug(f'**MSP: In data collate function')
    logger.debug(f'**MSP: Collating batch of size {len(batch)}')
    
    for query, text in batch:
        query_embedding = embed_model.get_query_embedding(query)
        text_embedding = embed_model.get_text_embedding(text)

        query_embeddings.append(torch.tensor(query_embedding))
        text_embeddings.append(torch.tensor(text_embedding))

    query_embeddings_t = torch.stack(query_embeddings)
    text_embeddings_t = torch.stack(text_embeddings)

    return query_embeddings_t, text_embeddings_t

def data_to_dataset(data):
    """
    Create a dataset consisting of (query, text) pairs from the data object
    """
    examples = []

    for query_id, query in data.queries.items():
        node_id = data.relevant_docs[query_id][0]
        text = data.corpus[node_id]

        examples.append((query, text))
    logger.info(f'**MSP: Number of examples: {len(examples)}')
    return examples


class MyDataset(Dataset):
    def __init__(self, data):
        self.queries, self.texts = self.get_examples(data)
        

    def get_examples(self, data):
        queries = []
        texts = []
        for query_id, query in data.queries.items():
            queries.append(query)

            node_id = data.relevant_docs[query_id][0]
            text = data.corpus[node_id]
            texts.append(text)
        logger.info(f'**MSP: MyDatset - Number of queries, texts: {len(queries)}, {len(texts)}')
        
        return queries, texts

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        return self.queries[idx], self.texts[idx]


