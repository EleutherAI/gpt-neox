import torch
import torch.nn as nn

from gpt_neox import print_rank_0
from gpt_neox.model.positional_embeddins import SinusoidalPositionalEmbedding


class Embedding(torch.nn.Module):
    def __init__(
        self,
        hidden_size,
        vocab_size,
        max_sequence_length,
        embedding_dropout_prob,
        init_method,
        num_tokentypes=0,
        pos_emb="learned",
        use_bnb_optimizer=False,
    ):
        super(Embedding, self).__init__()
        assert pos_emb in ["learned", "sinusoidal", None]
        self.hidden_size = hidden_size
        self.init_method = init_method
        self.num_tokentypes = num_tokentypes

        # Word embeddings (parallel).
        self.word_embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=self.hidden_size,
        )
        self.init_method(self.word_embeddings.weight)

        if use_bnb_optimizer:
            try:
                import bitsandbytes as bnb

                self.embedding_module = bnb.nn.StableEmbedding
            except ModuleNotFoundError:
                print(
                    "Please install bitsandbytes following https://github.com/facebookresearch/bitsandbytes."
                )
                raise Exception
        else:
            self.embedding_module = torch.nn.Embedding

        # Position embedding (serial).
        self.embedding_type = pos_emb
        if self.embedding_type == "learned":
            self.position_embeddings = self.embedding_module(
                max_sequence_length, self.hidden_size
            )
            self.init_method(self.position_embeddings.weight)
            # Initialize the position embeddings.
        elif self.embedding_type == "sinusoidal":
            self.position_embeddings = SinusoidalPositionalEmbedding(self.hidden_size)

        if self.num_tokentypes > 0:
            self.tokentype_embeddings = self.embedding_module(
                self.num_tokentypes, self.hidden_size
            )
            self.init_method(self.tokentype_embeddings.weight)
        else:
            self.tokentype_embeddings = None

        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)

    def add_tokentype_embeddings(self, num_tokentypes):
        if self.tokentype_embeddings is not None:
            raise Exception("tokentype embeddings is already initialized")

        print_rank_0("adding embedding for {} tokentypes".format(num_tokentypes))
        self.num_tokentypes = num_tokentypes
        self.tokentype_embeddings = self.embedding_module(
            num_tokentypes, self.hidden_size
        )
        # Initialize the token-type embeddings.
        self.init_method(self.tokentype_embeddings.weight)

    def forward(self, input_ids, position_ids, tokentype_ids=None):
        # Embeddings.
        words_embeddings = self.word_embeddings(input_ids)

        if self.embedding_type in ["learned", "sinusoidal"]:
            position_embeddings = self.position_embeddings(position_ids)
            embeddings = words_embeddings + position_embeddings
        else:
            embeddings = words_embeddings

        if tokentype_ids is not None:
            assert self.tokentype_embeddings is not None
            embeddings = embeddings + self.tokentype_embeddings(tokentype_ids)
        else:
            assert self.tokentype_embeddings is None

        return self.embedding_dropout(embeddings)
