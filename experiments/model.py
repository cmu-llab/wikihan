import torch
import torch.nn as nn
import torch.nn.functional as f
from collections import defaultdict


class Vocab:
    def __init__(self, vocabs):
        self._v2i = defaultdict(self._default_idx)
        v2i = {}
        for idx, vocab in enumerate(vocabs):
            v2i[vocab] = idx
        self._v2i.update(v2i)
        self._i2v = defaultdict(self._default_vocab)
        self._i2v.update({v: k for k, v in self._v2i.items()})
        self.default_idx = self._v2i['<unk>']
        self.len = len(self._v2i)
        assert len(self._v2i) == len(self._i2v)

    def _default_idx(self):
        return self.default_idx

    def _default_vocab(self):
        return '<unk>'

    def to_string(self, indices):
        # indices is a tensor
        str = ''
        for idx in indices:
            str += self._i2v[idx.item()]
        return str

    def __len__(self):
        # ensure length is static because defaultdict inserts unseen keys upon first access
        return self.len


class Embedding(nn.Module):
    """Feature extraction:
    Looks up embeddings for a source character and the language,
    concatenates the two,
    then passes through a linear layer
    """
    def __init__(self, embedding_dim, langs, C2I):
        super(Embedding, self).__init__()
        self.langs = langs
        self.char_embeddings = nn.Embedding(len(C2I), embedding_dim)
        self.lang_embeddings = nn.Embedding(len(self.langs), embedding_dim)
        # map concatenated source and language embedding to 1 embedding
        self.fc = nn.Linear(2 * embedding_dim, embedding_dim)

    def forward(self, char_indices, lang_indices):
        # both result in (L, E), where L is the length of the entire cognate set
        chars_embedded = self.char_embeddings(char_indices)
        lang_embedded = self.lang_embeddings(lang_indices)

        # concatenate the tensors to form one long embedding then map down to regular embedding size
        return self.fc(torch.cat((chars_embedded, lang_embedded), dim=-1))


class MLP(nn.Module):
    """
    Multi-layer perceptron to generate logits from the decoder state
    """
    def __init__(self, hidden_dim, feedforward_dim, output_size):
        # TODO: what are these magic numbers?
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, 2 * feedforward_dim)
        self.fc2 = nn.Linear(2 * feedforward_dim, feedforward_dim)
        self.fc3 = nn.Linear(feedforward_dim, output_size, bias=False)

    # no need to perform softmax because CrossEntropyLoss does the softmax for you
    def forward(self, decoder_state):
        h = f.relu(self.fc1(decoder_state))
        scores = self.fc3(f.relu(self.fc2(h)))
        return scores


class Attention(nn.Module):
    def __init__(self, hidden_dim, embedding_dim):
        # TODO: batch_first?
        super(Attention, self).__init__()
        self.W_query = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_c_s = nn.Linear(embedding_dim, hidden_dim, bias=False)

    def forward(self, query, keys, encoded_input):
        # query: decoder state. [1, 1, H]
        # keys: encoder states. [1, L, H]

        # TODO: why do we need this?
        query = self.W_query(query)
        # dot product attention to calculate similarity between the query and each key
        # scores: [1, L, 1]
        scores = torch.matmul(keys, query.transpose(1, 2))
        # TODO: do the softmax on the correct dimension
        # softmax to get a probability distribution over the L encoder states
        weights = f.softmax(scores, dim=-2)

        # TODO: do attention analysis and highlight the attention vector

        # weights: L x 1
        # encoded_input: L x E
        # keys: L x D
        # result: 1 x D - weighted version of the input
        weighted_states = weights * (self.W_c_s(encoded_input) + self.W_key(keys))
        weighted_states = weighted_states.sum(dim=-2)

        return weighted_states


class Model(nn.Module):
    """
    Encoder-decoder architecture
    """
    def __init__(self, C2I,
                 num_layers,
                 dropout,
                 feedforward_dim,
                 embedding_dim,
                 model_size,
                 model_type,
                 langs):
        super(Model, self).__init__()
        # TODO: modularize so we can get dialects for Austronesian, Chinese, or Romance
        # TODO: can we modularize this better?
        self.C2I = C2I

        # share embedding across all languages, including the proto-language
        self.embeddings = Embedding(embedding_dim, langs, C2I)
        # have separate embedding for the language
        # technically, most of the vocab is not used in the separator embedding
        # since the separator tokens and the character embeddings are disjoint, put them all in the same matrix

        self.langs = langs
        self.protolang = langs[0]
        self.L2I = {l: idx for idx, l in enumerate(langs)}

        self.dropout = nn.Dropout(dropout)
        if model_type == "gru":
            # TODO: beware of batching
            self.encoder_rnn = nn.GRU(input_size=embedding_dim,
                                      hidden_size=model_size,
                                      num_layers=num_layers,
                                      batch_first=True)
            self.decoder_rnn = nn.GRU(input_size=embedding_dim + model_size,
                                      hidden_size=model_size,
                                      num_layers=num_layers,
                                      batch_first=True)
        else:
            self.encoder_rnn = nn.LSTM(input_size=embedding_dim,
                                       hidden_size=model_size,
                                       num_layers=num_layers,
                                       batch_first=True)
            self.decoder_rnn = nn.LSTM(input_size=embedding_dim + model_size,
                                       hidden_size=model_size,
                                       num_layers=num_layers,
                                       batch_first=True)

        # TODO: shouldn't we share this with the encoder?

        self.mlp = MLP(hidden_dim=model_size, feedforward_dim=feedforward_dim, output_size=len(C2I))
        self.attention = Attention(hidden_dim=model_size, embedding_dim=embedding_dim)

    def forward(self, source_tokens, source_langs, target_tokens, target_langs, device):
        # encoder
        # TODO: is the encoder treating each input as separate?
        # encoder_states: 1 x L x H, memory: 1 x 1 x H, where L = len(daughter_forms)
        (encoder_states, memory), embedded_cognateset = self.encode(source_tokens, source_langs, device)
        # perform dropout on the output of the RNN
        encoder_states = self.dropout(encoder_states)

        # decoder
        # start of protoform sequence
        # TODO: is this really necessary? we already have < and > serving as BOS/EOS
        start_char = (torch.tensor([self.C2I._v2i["<s>"]]).to(device), torch.tensor([self.L2I["sep"]]).to(device))
        start_encoded = self.embeddings(*start_char)
        # initialize weighted states to the final encoder state
        # TODO: there has to be a better way of doing this indexing - preserve batch dim
        attention_weighted_states = memory.squeeze(dim=0)
        # start_encoded: 1 x E, attention_weighted_states: 1 x H
        # concatenated into 1 x (H + E)
        decoder_input = torch.cat((start_encoded, attention_weighted_states), dim=1).unsqueeze(dim=0)
        # perform dropout on the input to the RNN
        decoder_input = self.dropout(decoder_input)
        decoder_state, _ = self.decoder_rnn(decoder_input)
        # perform dropout on the output of the RNN
        decoder_state = self.dropout(decoder_state)
        scores = []  # TODO: it's faster to initialize the shape then fill it in

        # TODO: could we even do this batched or pass in the whole target?? but then we don't control the attention
            # there is a batched way of doing it

        for lang, char in zip(target_langs, target_tokens):
            # lang will either be sep or the protolang
            # embedding layer
            true_char_embedded = self.embeddings(char, lang).unsqueeze(dim=0)
            # MLP to get a probability distribution over the possible output phonemes
            char_scores = self.mlp(decoder_state + attention_weighted_states)
            scores.append(char_scores.squeeze(dim=0))
            # dot product attention over the encoder states - results in (1, H)
            attention_weighted_states = self.attention(decoder_state, encoder_states, embedded_cognateset)
            # decoder_input: (1, 1, H + E)
            decoder_input = torch.cat((true_char_embedded, attention_weighted_states), dim=1).unsqueeze(dim=0)
            # TODO: make sure that we're really taking the decoder state

            # perform dropout on the input to the RNN
            decoder_input = self.dropout(decoder_input)
            decoder_state, _ = self.decoder_rnn(decoder_input)
            # perform dropout on the output of the RNN
            decoder_state = self.dropout(decoder_state)

        # |T| elem list with (1, |Y|) -> (T, |Y|)
        scores = torch.vstack(scores)
        return scores

    def encode(self, source_tokens, source_langs, device):
        # daughter_forms: list of lang and indices in the vocab
        embedded_cognateset = self.embeddings(source_tokens, source_langs).to(device)
        # batch size of 1
        embedded_cognateset = embedded_cognateset.unsqueeze(dim=0)

        # TODO: note that the LSTM returns something diff than the GRU in pytorch
        # perform dropout on the input to the RNN
        embedded_cognateset = self.dropout(embedded_cognateset)
        return self.encoder_rnn(embedded_cognateset), embedded_cognateset

    def decode(self, encoder_states, memory, embedded_cognateset, max_length, device):
        # greedy decoding - generate protoform by picking most likely sequence at each time step

        start_char = (torch.tensor([self.C2I._v2i["<s>"]]).to(device), torch.tensor([self.L2I["sep"]]).to(device))
        start_encoded = self.embeddings(*start_char).to(device)

        # initialize weighted states to the final encoder state
        # TODO: there has to be a better way of doing this indexing - preserve batch dim
        attention_weighted_states = memory.squeeze(dim=0)
        # start_encoded: 1 x E, attention_weighted_states: 1 x H
        # concatenated into 1 x (H + E)
        decoder_input = torch.cat((start_encoded, attention_weighted_states), dim=1).unsqueeze(dim=0)
        decoder_state, _ = self.decoder_rnn(decoder_input)
        # TODO: is there a better way to do this?
        reconstruction = []

        i = 0
        while i < max_length:
            # embedding layer
            # MLP to get a probability distribution over the possible output phonemes
            char_scores = self.mlp(decoder_state + attention_weighted_states)
            # TODO: make sure it's along the correct dimension
            # char_scores: [1, 1, |Y|]
            predicted_char = torch.argmax(char_scores.squeeze(dim=0)).item()
            predicted_char_idx = predicted_char

            predicted_char = (torch.tensor([predicted_char]).to(device), torch.tensor([self.L2I[self.protolang]]).to(device))
            predicted_char_embedded = self.embeddings(*predicted_char)

            # dot product attention over the encoder states
            attention_weighted_states = self.attention(decoder_state, encoder_states, embedded_cognateset)
            # (1, 1, H + E)
            decoder_input = torch.cat((predicted_char_embedded, attention_weighted_states), dim=1).unsqueeze(dim=0)
            decoder_state, _ = self.decoder_rnn(decoder_input)

            reconstruction.append(predicted_char_idx)

            i += 1
            # end of sequence generated
            # TODO: declare EOS as a global variable - same with BOS
            if predicted_char_idx == self.C2I._v2i[">"]:
                break

        return torch.tensor(reconstruction)
