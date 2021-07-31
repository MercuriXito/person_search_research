import torch
import torch.nn as nn
import torch.nn.init as init


class ReIDEmbeddingHead(nn.Module):

    def __init__(self, featmap_names=['feat_res5'],
                 in_channels=[2048],
                 dim=256,
                 feature_norm=True):
        super(ReIDEmbeddingHead, self).__init__()
        self.featmap_names = featmap_names
        self.in_channels = list(map(int, in_channels))
        self.dim = int(dim)

        self.projectors = nn.ModuleDict()
        indv_dims = self._split_embedding_dim()
        for ftname, in_chennel, indv_dim in zip(self.featmap_names, self.in_channels, indv_dims):
            indv_dim = int(indv_dim)
            proj = nn.Sequential(
                nn.Linear(int(in_chennel), indv_dim),
                nn.BatchNorm1d(indv_dim))
            init.normal_(proj[0].weight, std=0.01)
            init.normal_(proj[1].weight, std=0.01)
            init.constant_(proj[0].bias, 0)
            init.constant_(proj[1].bias, 0)
            self.projectors[ftname] = proj

        self.rescaler = nn.BatchNorm1d(1, affine=True)
        self.feature_norm = feature_norm

    def forward(self, featmaps):
        '''
        Arguments:
            featmaps: OrderedDict[Tensor], and in featmap_names you can choose which
                      featmaps to use
        Returns:
            tensor of size (BatchSize, dim), L2 normalized embeddings.
            tensor of size (BatchSize, ) rescaled norm of embeddings, as class_logits.
        '''
        if len(featmaps) == 1:
            k, v = list(featmaps.items())[0]
            v = self._flatten_fc_input(v)
            embeddings = self.projectors[k](v)
            if self.feature_norm:
                norms = embeddings.norm(2, 1, keepdim=True)
                embeddings = embeddings / \
                    norms.expand_as(embeddings).clamp(min=1e-12)
                norms = self.rescaler(norms)
            else:
                norms = torch.zeros_like(embeddings.norm(2, 1, keepdim=True))
            return embeddings, norms
        else:
            outputs = []
            for k, v in featmaps.items():
                v = self._flatten_fc_input(v)
                outputs.append(
                    self.projectors[k](v)
                )
            embeddings = torch.cat(outputs, dim=1)
            if self.feature_norm:
                norms = embeddings.norm(2, 1, keepdim=True)
                embeddings = embeddings / \
                    norms.expand_as(embeddings).clamp(min=1e-12)
                norms = self.rescaler(norms)
            else:
                norms = torch.zeros_like(embeddings.norm(2, 1, keepdim=True))
            return embeddings, norms

    @property
    def rescaler_weight(self):
        return self.rescaler.weight.item()

    def _flatten_fc_input(self, x):
        if x.ndimension() == 4:
            assert list(x.shape[2:]) == [1, 1]
            return x.flatten(start_dim=1)
        return x  # ndim = 2, (N, d)

    def _split_embedding_dim(self):
        parts = len(list(self.in_channels))
        tmp = [self.dim / parts] * parts
        if sum(tmp) == self.dim:
            return tmp
        else:
            res = self.dim % parts
            for i in range(1, res + 1):
                tmp[-i] += 1
            assert sum(tmp) == self.dim
            return tmp
