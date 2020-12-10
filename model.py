import tensorflow as tf
from tensorflow.keras import layers,models
import hyperparameters as hp
from helpers import create_embed_layer

NUM_CLASSES = 8

class KerasModel(models.Model):
    def __init__(self, obj, vocab_size, embedding_dim, elmo_embedding_dim, hidden_dim, num_layers):
        super(KerasModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.elmo_embedding_dim = elmo_embedding_dim

        # embedding layers
        self.sentence_embeddings, _ = create_embed_layer(obj, vocab_size, 0)
        self.context_embeddings, _ = create_embed_layer(obj, vocab_size, 0)

        #LSTM layers
        self.LSTM1 = layers.Bidirectional(layers.LSTM(self.hidden_dim, num_layers=self.num_layers, dropout=hp.enc_dropout))
        self.LSTM2 = layers.Bidirectional(layers.LSTM(self.hidden_dim, num_layers=self.num_layers, dropout=hp.enc_dropout))

        self.dropout = layers.Dropout(hp.dropout)
        self.classifier = layers.Dense(NUM_CLASSES)


    def forward(self, input_sent, input_sent_ids, input_char, input_char_ids,sent_lens, ctx_lens, input_label, input_label_ids):
        # sort sentences
        _, sent_perm_idx = sent_lens.sort(0, descending=True)
        input_sent_ids = input_sent_ids[sent_perm_idx]

        #sort context
        _, ctx_perm_idx = ctx_lens.sort(0, descending=True)
        input_char_ids = input_char_ids[ctx_perm_idx]

        self.x_sent_ids = self.sentence_embeddings(input_sent_ids)
        self.x_ctx_ids = self.context_embeddings(input_char_ids)

        # can skip packing I think, keras layers dont take a set input length?
        _, (hn_sent, _) = self.LSTM1(self.x_sent_ids)
        _, (hn_ctx, _) = self.LSTM2(self.x_ctx_ids)

        hn_sent = hn_sent.reshape(self.num_layers, 2, -1, self.hidden_dim)[1]
        hn_ctx = hn_ctx.reshape(self.num_layers, 2, -1, self.hidden_dim)[1]

        encoder_sent = tf.concat((hn_sent[0], hn_sent[1]), 1)
        encoder_ctx = tf.concat((hn_ctx[0], hn_ctx[1]), 1)

        out_sent_ctx = tf.concat((encoder_sent, encoder_ctx), 1)


        return self.classifier(self.dropout(out_sent_ctx))
        