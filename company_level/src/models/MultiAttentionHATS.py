import numpy as np
import os
import random
import re
import time

# from base.base_model import BaseModel
import tensorflow as tf


class MultiHeadHATS():
    def __init__(self, config):
        super(MultiHeadHATS, self).__init__(config)
        self.num_labels = len(config.label_proportion)
        self.input_dimension = len(config.feature_list)
        self.num_layers = config.num_layer
        self.dropout_keep_prob = 1 - config.dropout
        self.max_gradient_norm = config.grad_max_norm
        self.num_relations = config.num_relations
        self.node_feature_size = config.node_feat_size
        self.relation_projection = config.rel_projection
        self.feature_attention = config.feat_att
        self.relation_attention = config.rel_att
        self.attention_heads = config.att_heads  # For multi-head attention
        self.top_k_attention = config.att_topk
        self.num_companies = config.num_companies
        self.sampled_neighbors = config.neighbors_sample

        self.build_model()
        self.init_saver()

    def get_node_state(self, state_module):
        if state_module == 'lstm':
            lstm_cells = [tf.contrib.rnn.BasicLSTMCell(self.node_feature_size) for _ in range(self.num_layers)]
            dropout_layers = [tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=self.dropout_keep_prob,
                                                            output_keep_prob=self.dropout_keep_prob) for cell in lstm_cells]
            lstm_stack = tf.nn.rnn_cell.MultiRNNCell(dropout_layers, state_is_tuple=True)
            lstm_outputs, lstm_state = tf.nn.dynamic_rnn(lstm_stack, self.input_placeholder, dtype=tf.float32)
            lstm_state_padded = tf.concat([tf.zeros([1, lstm_state[-1][-1].shape[1]]), lstm_state[-1][-1]], 0)
        return lstm_state_padded

    def relation_projection(self, state, relation_idx):
        with tf.variable_scope('relation_projection_' + str(relation_idx)):
            projected_relation_state = tf.layers.dense(inputs=state, units=self.node_feature_size,
                                                       activation=tf.nn.leaky_relu, name='relation_projection')
        return projected_relation_state

    def multi_head_attention(self, node_features, neighbor_features, head_count, relation_idx):
        with tf.variable_scope('multi_head_attention_' + str(relation_idx)):
            attention_scores = []
            for head in range(head_count):
                head_features = tf.layers.dense(inputs=neighbor_features, units=self.node_feature_size,
                                                activation=tf.nn.leaky_relu, name=f'attention_head_{head}')
                combined_features = tf.concat([node_features, head_features], axis=-1)
                attention_weights = tf.layers.dense(inputs=combined_features, units=1, name=f'attention_score_{head}')
                attention_scores.append(tf.nn.softmax(attention_weights, axis=1))

            # Concatenating multi-head attention scores
            concatenated_attention_scores = tf.concat(attention_scores, axis=-1)
        return concatenated_attention_scores

    # Example Usage


    def relation_attention(self, node_state):
        with tf.variable_scope('relation_attention'):
            neighbor_embeddings = tf.nn.embedding_lookup(node_state, self.relation_matrix)
            expanded_state = tf.expand_dims(tf.expand_dims(node_state[1:], 1), 0)
            expanded_state = tf.tile(expanded_state, [self.num_relations, 1, self.max_k_neighbors, 1])

            concatenated_embeddings = tf.concat([neighbor_embeddings, expanded_state], -1)
            multi_attention_scores = self.multi_head_attention(neighbor_embeddings, concatenated_embeddings,
                                                               self.attention_heads, 'relation_attention')
            aggregated_relation_representation = tf.reduce_sum(neighbor_embeddings * multi_attention_scores, axis=2) / \
                                                 tf.expand_dims((tf.to_float(self.relation_count) + 1e-10), -1)
        return aggregated_relation_representation

    def aggregate_relation_embeddings(self):
        def project_relation_embeddings(embedding):
            projected_embeddings = []
            for i in range(embedding.shape[0]):
                expanded_embedding = tf.tile(tf.expand_dims(embedding[i], 0), [self.num_companies, 1])
                projected_embeddings.append(tf.expand_dims(expanded_embedding, 0))
            return tf.concat(projected_embeddings, 0)

        with tf.name_scope('aggregate_relation_embeddings'):
            relation_embedding = project_relation_embeddings(self.relation_embedding)

            concatenated_relation_embedding = tf.concat([self.aggregated_relation_representation, relation_embedding], -1)
            multi_attention_scores = self.multi_head_attention(self.aggregated_relation_representation,
                                                               concatenated_relation_embedding, self.attention_heads,
                                                               'aggregate_relation')
            updated_node_state = tf.reduce_mean(self.aggregated_relation_representation * multi_attention_scores, axis=0)
        return updated_node_state

    def build_model(self):
        self.dropout_keep_prob_placeholder = tf.placeholder_with_default(1.0, shape=())
        self.input_placeholder = tf.placeholder(tf.float32, shape=[None, self.config.lookback, self.input_dimension])
        self.label_placeholder = tf.placeholder(tf.float32, shape=[None, self.num_labels])
        self.relation_matrix = tf.placeholder(tf.int32, shape=[None, None, self.sampled_neighbors])
        self.relation_count = tf.placeholder(tf.int32, shape=[None, None])
        self.max_k_neighbors = tf.placeholder(tf.int32, shape=())

        self.relation_embedding = self.create_relation_onehot()

        lstm_state = self.get_node_state('lstm')
        self.aggregated_relation_representation = self.relation_attention(lstm_state)
        aggregated_relation_summary = self.aggregate_relation_embeddings()
        updated_node_state = aggregated_relation_summary + lstm_state[1:]

        logits = tf.layers.dense(inputs=updated_node_state, units=self.num_labels,
                                 activation=tf.nn.leaky_relu, name='final_prediction')
        self.probabilities = tf.nn.softmax(logits)
        self.predictions = tf.argmax(logits, axis=-1)

        with tf.name_scope("loss"):
            self.cross_entropy_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.label_placeholder, logits=logits))
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            total_loss = self.cross_entropy_loss
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = tf.train.AdamOptimizer(self.config.lr).minimize(total_loss,
                                                                                  global_step=self.global_step_tensor)

            correct_predictions = tf.equal(tf.argmax(logits, axis=-1), tf.argmax(self.label_placeholder, axis=-1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    def init_saver(self):
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)



class SectorWiseGraph:
        def __init__(self, csv_file, embedding_dim=128):
            self.data = pd.read_csv(csv_file)

            # Ensure that the data contains 'Sector' and 'Symbol' columns and precomputed embeddings.
            assert 'Sector' in self.data.columns and 'Symbol' in self.data.columns

            self.sector_column = 'Sector'
            self.symbol_column = 'Symbol'
            self.sector_labels = self.data[self.sector_column].unique()


            self.embedding_dim = embedding_dim
            self.company_embeddings = self._generate_embeddings()

            # Graph-related data
            self.node_features = []
            self.edge_index = []
            self.sector_mappings = []  # Stores the sector information for each company (node)
            self.node_idx = 0  # Index to keep track of node numbers across sectors

        def _generate_embeddings(self):
            """ Simulate the embeddings for each company (from LSTM )"""
            company_embeddings = {}
            for idx, company in enumerate(self.data[self.symbol_column].unique()):
                # Generate a random embedding for each company
                company_embeddings[company] = torch.randn(self.embedding_dim)
            return company_embeddings

        def create_sector_graph(self):
            """ This function creates a fully connected graph for each sector. """
            for sector in self.sector_labels:
                print(f"Building graph for sector: {sector}")

                # Filter data for this sector
                sector_companies = self.data[self.data[self.sector_column] == sector][self.symbol_column].unique()
                num_companies = len(sector_companies)

                # Build fully connected graph (clique) for the sector
                G = nx.complete_graph(num_companies)  # Fully connected graph
                sector_edges = from_networkx(G).edge_index  # Get edge_index in PyG format

                # Adjust node indices to reflect the entire dataset
                sector_edges += self.node_idx
                self.edge_index.append(sector_edges)

                # Add node features (embeddings) and track sector mappings
                for company in sector_companies:
                    self.node_features.append(self.company_embeddings[company])
                    self.sector_mappings.append(sector)

                # Update node index to match the next set of nodes (companies)
                self.node_idx += num_companies

        def get_graph_data(self):
            """ Return the graph data as a PyG Data object. """
            self.create_sector_graph()

            # Convert node features to tensor and concatenate all sector graphs' edge indices
            node_features_tensor = torch.stack(self.node_features)
            edge_index_tensor = torch.cat(self.edge_index, dim=1)

            # Construct PyTorch Geometric Data object
            data = Data(x=node_features_tensor, edge_index=edge_index_tensor)
            return data


