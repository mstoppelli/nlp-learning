import tensorflow as tf
import hyperparameters as hp
import numpy as np

def create_embed_layer(obj, vocab_size, padding_idx, glove_dict):
    masked_data = ["<start>", "<end>", "<pad>"]
    num_embeddings = vocab_size + 1   ##+1 to take into account the unk token
    embedding_dim = hp.embedding_dim

    padding_idx = tf.constant([padding_idx])

    embedding_layer = tf.keras.layers.Embedding(num_embeddings, embedding_dim, mask_zero=True)

    emb_dict_pre_train = dict() ##to get the embeddings before training

    for word, i in obj.term2index.items():

        if word in masked_data:
            new_weights = embedding_layer.weights()
            new_weights[i] = 0
            embedding_layer.set_weights(new_weights)
            emb_dict_pre_train[word] = embedding_layer.weights()[i]
            continue

        elif word in glove_dict:
            vec = tf.constant(glove_dict[word])
            new_weights = embedding_layer.weights()
            new_weights[i] = vec
            embedding_layer.set_weights(new_weights)
            emb_dict_pre_train[word] = glove_dict[word]

        else:
            ##for words not present in the glove dictionary
            new_weights = embedding_layer.weights()
            new_weights[i] = tf.convert_to_tensor(np.random.uniform(-0.2, 0.2, embedding_dim))
            embedding_layer.set_weights(new_weights)
            emb_dict_pre_train[word] = embedding_layer.weights()[i]  
    

    return embedding_layer, num_embeddings

