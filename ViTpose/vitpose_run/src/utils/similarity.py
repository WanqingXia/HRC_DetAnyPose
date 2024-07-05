from torch import nn
import torch.nn.functional as F


class CosineSimilarity(nn.Module):
    def __init__(self):
        super(CosineSimilarity, self).__init__()

    def forward(self, query, reference):
        # Compute the cosine similarity between the query and reference embeddings
        # query dimention must be [1, N]
        query = F.normalize(query, dim=-1)
        reference = F.normalize(reference, dim=-1)
        # Expand the query to match the batch size of reference
        query_expanded = query.expand(reference.size(0), -1)

        similarity = F.cosine_similarity(query_expanded, reference, dim=-1)
        return similarity.clamp(min=0.0, max=1.0)


