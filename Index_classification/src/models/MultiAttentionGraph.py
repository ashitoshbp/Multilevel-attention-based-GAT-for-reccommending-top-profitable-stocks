import tensorflow as tf
import numpy as np
import os
import random
import re
import time

# from config import get_args
# from base.base_model import BaseModel


class GraphAttentionNetwork():
    def __init__(self, config):
        super(GraphAttentionNetwork, self).__init__(config)
        self.model_type = config.GNN_model
        self.input_feature_dim = len(config.feature_list)
        self.dropout_rate = 1 - config.dropout
        self.node_feature_count = config.node_features
        self.use_bias_term = config.use_bias
        self.stack_layers_enabled = config.stack_layer
        self.layer_count = config.num_layer
        self.num_classes = len(config.label_proportion)
        self.num_relationships = config.num_relations
        self.num_entities = config.num_companies

        self.build_graph()
        self.initialize_saver()

    def extract_feature_state(self, model_type=None):
        with tf.variable_scope('feature_extraction_ops'):
            if 'LSTM' in model_type:
                lstm_layers = [tf.contrib.rnn.BasicLSTMCell(64, state_is_tuple=True) for _ in range(self.layer_count)]
                lstm_layers_with_dropout = [tf.contrib.rnn.DropoutWrapper(layer, input_keep_prob=self.dropout_rate,
                                                                          output_keep_prob=self.dropout_rate) for layer
                                            in lstm_layers]
                lstm_cell = tf.nn.rnn_cell.MultiRNNCell(lstm_layers_with_dropout, state_is_tuple=True)
                lstm_output, lstm_state = tf.nn.dynamic_rnn(lstm_cell, self.input_features, dtype=tf.float32)
                return lstm_state[-1][-1]

    def generate_relation_embeddings(self):
        relation_embeddings = []
        for relationship_index in range(self.num_relationships):
            relation_embeddings.append(tf.one_hot([relationship_index], depth=self.num_relationships))
        return tf.concat(relation_embeddings, 0)

    def matrix_multiplication(self, matrix_a, matrix_b, is_sparse=False):
        if is_sparse:
            return tf.sparse_tensor_dense_matmul(matrix_a, matrix_b)
        return tf.matmul(matrix_a, matrix_b)

    def update_node_features(self, current_node_features, layer_number=0):
        if self.model_type == 'GAT_Model':
            def adjust_input_shape(embeddings):
                reshaped_embeddings = []
                for i in range(embeddings.shape[0]):
                    expanded_embedding = tf.tile(tf.expand_dims(tf.expand_dims(embeddings[i], 0), 0),
                                                 [self.num_entities, 20, 1])
                    reshaped_embeddings.append(tf.expand_dims(expanded_embedding, 0))
                return tf.concat(reshaped_embeddings, 0)

            with tf.variable_scope('graph_operations'):
                current_node_features = tf.concat(
                    [tf.zeros([1, current_node_features.shape[1]]), current_node_features], 0)
                neighbor_nodes = tf.nn.embedding_lookup(current_node_features, self.relationship_matrix)
                expanded_node_state = tf.expand_dims(tf.expand_dims(current_node_features[1:], 1), 0)
                expanded_node_state = tf.tile(expanded_node_state, [self.num_relationships, 1, 20, 1])
                reshaped_relation_embeddings = adjust_input_shape(self.relation_embeddings)

                attention_input = tf.concat([neighbor_nodes, expanded_node_state, reshaped_relation_embeddings], -1)
                attention_scores = tf.layers.dense(inputs=attention_input, units=1, name='attention_weights')
                mask_for_attention = tf.to_float(tf.expand_dims(tf.sequence_mask(self.relationship_count, 20), -1))
                normalized_attention_scores = tf.nn.softmax(attention_scores, 2)
                weighted_neighbor_sum = tf.reduce_sum(neighbor_nodes * normalized_attention_scores, 2)
                relation_representation = weighted_neighbor_sum / tf.expand_dims(
                    (tf.to_float(self.relationship_count) + 1e-10), -1)
                updated_node_state = tf.reduce_mean(relation_representation, 0)
            return updated_node_state

    def apply_pooling(self, node_states, pooling_type):
        if pooling_type == 'mean':
            pooled_state = tf.reduce_mean(node_states, 0)
        elif pooling_type == 'max':
            pooled_state = tf.reduce_max(node_states, 0)
        return pooled_state

    def build_graph(self):
        self.dropout_rate_placeholder = tf.placeholder_with_default(1.0, shape=())
        self.input_features = tf.placeholder(tf.float32, shape=[None, self.config.lookback, self.input_feature_dim])
        self.target_labels = tf.placeholder(tf.float32, shape=[None, 3])

        if self.model_type == 'GAT_Model':
            self.relationship_matrix = tf.placeholder(tf.int32, shape=[None, None, None])
            self.relationship_count = tf.placeholder(tf.int32, shape=[None, None])
            self.relation_embeddings = self.generate_relation_embeddings()

        node_feature_state = self.extract_feature_state(model_type=self.config.price_model)
        self.global_node_state = node_feature_state[0]
        entity_node_states = node_feature_state[1:]

        self.updated_node_features = self.update_node_features(entity_node_states)

        pooled_node_states = self.apply_pooling(entity_node_states, 'max')
        combined_state = tf.expand_dims(tf.concat([self.global_node_state, pooled_node_states], 0), 0)

        logits = tf.layers.dense(inputs=combined_state, units=3, name='final_output', activation=tf.nn.leaky_relu)

        self.probabilities = tf.nn.softmax(logits)
        self.predicted_labels = tf.argmax(logits, -1)

        with tf.name_scope("loss_computation"):
            self.loss_value = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.target_labels, logits=logits))
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.training_step = tf.train.AdamOptimizer(self.config.lr).minimize(self.loss_value,
                                                                                     global_step=self.global_step_tensor)

            correct_predictions = tf.equal(self.predicted_labels, tf.argmax(self.target_labels, -1))
            self.accuracy_metric = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    def initialize_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)


def HATS():
    return None