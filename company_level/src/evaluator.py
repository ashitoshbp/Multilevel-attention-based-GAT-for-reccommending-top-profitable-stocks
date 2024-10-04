import math
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, f1_score


class Evaluator:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.n_labels = len(config.label_proportion)

    def sample_neighbors(self, data):
        k = self.config.neighbors_sample
        neighbors_batch = []

        if self.config.model_type == 'HATS':
            for rel_neighbors in data.neighbors:
                rel_neighbors_batch = []
                for cpn_idx, neighbors in enumerate(rel_neighbors):
                    if neighbors.shape[0] < k:
                        shortfall = k - neighbors.shape[0]
                        neighbors = np.concatenate([neighbors, np.zeros(shortfall)])
                    else:
                        neighbors = np.random.choice(neighbors, k)
                    rel_neighbors_batch.append(np.expand_dims(neighbors, 0))

                neighbors_batch.append(np.expand_dims(np.concatenate(rel_neighbors_batch, axis=0), 0))

        return np.concatenate(neighbors_batch, axis=0)

    def get_rel_multi_hot(self, batch_neighbors, data):
        neighbors_multi_hot = []

        for cpn_idx, neighbors in enumerate(batch_neighbors):
            multi_hots = []
            for n_i in neighbors:
                if n_i == 0:
                    multi_hots.append(np.expand_dims(data.rel_multi_hot[cpn_idx, cpn_idx], axis=0))
                else:
                    multi_hots.append(np.expand_dims(data.rel_multi_hot[cpn_idx, int(n_i)], axis=0))
            neighbors_multi_hot.append(np.expand_dims(np.concatenate(multi_hots, axis=0), 0))

        return np.concatenate(neighbors_multi_hot, axis=0)

    def create_feed_dict(self, model, data, x, y, phase, neighbors=None):
        feed_dict = {
            model.x: x,
            model.y: y,
            model.rel_num: data.rel_num,
            model.rel_mat: neighbors,
            model.max_k: neighbors.shape[-1]
        }
        return feed_dict

    def get_result(self, sess, model, data, phase, neighbors=None):
        all_x, all_y, all_rt = next(data.get_batch(phase, self.config.lookback))
        preds, probs = [], []

        for x, y, rt in zip(all_x, all_y, all_rt):
            feed_dict = self.create_feed_dict(model, data, x, y, phase, neighbors)
            pred, prob = sess.run([model.prediction, model.prob], feed_dict=feed_dict)
            preds.append(pred)
            probs.append(prob)

        return np.array(preds), np.array(probs)

    def evaluate(self, sess, model, data, phase, neighbors=None):
        all_x, all_y, all_rt = next(data.get_batch(phase, self.config.lookback))
        losses, metrics = [], []

        for x, y, rt in zip(all_x, all_y, all_rt):
            feed_dict = self.create_feed_dict(model, data, x, y, phase, neighbors)
            loss, pred, prob = sess.run([model.cross_entropy, model.prediction, model.prob], feed_dict=feed_dict)
            label = np.argmax(y, axis=1)
            metrics_all, metrics_topk = self.metric(label, pred, prob, rt)

            losses.append(loss)
            metrics.append(metrics_all)

        avg_loss = np.mean(losses)
        avg_metrics = {key: np.mean([m[key] for m in metrics], axis=0) for key in metrics[0]}
        return avg_loss, avg_metrics

    def create_confusion_matrix(self, y, y_, is_distribution=False):
        n_samples = float(y_.shape[0])

        if is_distribution:
            label_ref = np.argmax(y_, axis=1)
            label_hyp = np.argmax(y, axis=1)
        else:
            label_ref, label_hyp = y, y_

        p_in_hyp = np.sum(label_hyp)
        n_in_hyp = n_samples - p_in_hyp
        tp = np.sum(label_ref * label_hyp)
        fp = p_in_hyp - tp
        tn = n_samples - np.count_nonzero(label_ref + label_hyp)
        fn = n_in_hyp - tn

        return float(tp), float(fp), float(tn), float(fn)

    def get_mcc(self, tp, fp, tn, fn):
        denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        return (tp * tn - fp * fn) / math.sqrt(denominator) if denominator else 0

    def get_f1(self, tp, fp, tn, fn):
        eps = 1e-10
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        return 2 * (precision * recall) / (precision + recall + eps)

    def get_f1(self, y, y_):
        return f1_score(y, y_, average='macro'), f1_score(y, y_, average='micro')

    def get_acc(self, conf_mat):
        accuracy = conf_mat.trace() / conf_mat.sum()

        if self.n_labels == 2:
            compact_accuracy = accuracy
        else:
            compact_conf_mat = np.take(conf_mat, [[0, 2], [6, 8]])
            compact_accuracy = compact_conf_mat.trace() / compact_conf_mat.sum()

        return accuracy, compact_accuracy
    def attention_score(self):
        import tensorflow as tf

        class AttentionLayer(tf.keras.layers.Layer):
            def __init__(self, units):
                super(AttentionLayer, self).__init__()
                self.units = units

            def build(self, input_shape):
                # Initialize weights for node-level attention
                self.node_weights = self.add_weight(
                    shape=(input_shape[-1], self.units),
                    initializer='random_normal',
                    trainable=True,
                    name='node_weights'
                )
                # Initialize weights for graph-level attention
                self.graph_weights = self.add_weight(
                    shape=(self.units, 1),
                    initializer='random_normal',
                    trainable=True,
                    name='graph_weights'
                )

            def call(self, inputs):
                # Node-level attention
                node_attention = tf.nn.softmax(tf.matmul(inputs, self.node_weights), axis=1)

                # Calculate attention scores between nodes (fully connected graph)
                node_scores = tf.matmul(node_attention, tf.transpose(node_attention, perm=[0, 2, 1]))

                # Graph-level attention: aggregate node scores
                graph_attention = tf.reduce_mean(tf.matmul(node_attention, self.graph_weights), axis=1)

                # Final attention score (combining node and graph scores)
                final_attention_scores = node_scores * graph_attention[:, tf.newaxis, tf.newaxis]

                return final_attention_scores, node_scores, graph_attention

        # Example usage:
        inputs = tf.random.normal((batch_size, num_nodes, feature_dim))  # Replace with actual input tensor
        attention_layer = AttentionLayer(units=64)
        final_attention, node_attention, graph_attention = attention_layer(inputs)


    def expected_return(self, pred, prob, returns):
        n_mid = prob.shape[0] // 2
        short_half_idx = np.argsort(prob[:, 0])[-n_mid:]
        long_half_idx = np.argsort(prob[:, -1])[-n_mid:]
        short_rts = (returns[short_half_idx] * (pred[short_half_idx] == 0)).mean() * -1
        long_rts = (returns[long_half_idx] * (pred[long_half_idx] == (self.n_labels - 1))).mean()

        return (short_rts + long_rts) * 100

    def filter_topk(self, label, pred, prob, returns, topk):
        short_k_idx = np.argsort(prob[:, 0])[-topk:]
        long_k_idx = np.argsort(prob[:, -1])[-topk:]
        topk_idx = np.concatenate([short_k_idx, long_k_idx])

        return label[topk_idx], pred[topk_idx], prob[topk_idx], returns[topk_idx]

    def cal_metric(self, label, pred, prob, returns):
        exp_returns = self.expected_return(pred, prob, returns)
        conf_mat = confusion_matrix(label, pred, labels=[i for i in range(self.n_labels)])
        acc, cpt_acc = self.get_acc(conf_mat)
        mac_f1, mic_f1 = self.get_f1(label, pred)
        pred_rate = [(pred == i).sum() / pred.shape[0] for i in range(self.n_labels)]

        return {
            'pred_rate': pred_rate,
            'acc': acc,

        }

    def metric(self, label, pred, prob, returns, topk=30):
        metric_all = self.cal_metric(label, pred, prob, returns)
        label, pred, prob, returns = self.filter_topk(label, pred, prob, returns, topk)
        metric_topk = self.cal_metric(label, pred, prob, returns)

        return metric_all, metric_topk
