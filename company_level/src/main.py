import os
import time
import random
import itertools
from temp import randomize
import tensorflow as tf
# import tensorflow.contrib.slim as slim
from tensorflow.python.ops import nn
from torch.utils.data import dataloader

from Index_classification.src.models.MultiAttentionGraph import HATS
from logger import set_logger
from config import get_args
from trainer import Trainer
from evaluator import Evaluator


def init_prediction_model(config):
    with tf.variable_scope("model"):
        if config.model_type == "HATS":
            model = HATS(config)
    return model



class StockDataset:
    pass


class LSTMNetwork:
    pass

def main():
    simulate_long_running_process(3)
    config = get_args()
    logger = set_logger(config)

    class StockDataset(Dataset):
        def __init__(self, csv_file, sequence_length=30):
            self.data = pd.read_csv(csv_file)
            self.sequence_length = sequence_length
            self.feature_columns = [
                'Open', 'High', 'Low', 'Close', 'Volume', 'nor_close',
                'return ratio', 'c_open', 'c_high', 'c_low',
                '5-days', '10-days', '15-days', '20-days', '25-days', '30-days',
                'label_Utilities', 'label_Basic Materials', 'label_Financial Services',
                'label_Consumer Cyclical', 'label_Communication Services',
                'label_Energy', 'label_Consumer Defensive', 'label_Healthcare',
                'label_Technology', 'label_Industrials'
            ]

            self.symbols = self.data['Symbol'].unique()

            # Normalize numeric features (optional)
            self.data[self.feature_columns] = self.data[self.feature_columns].apply(lambda x: (x - x.mean()) / x.std())

        def __len__(self):
            return len(self.data) - self.sequence_length

        def __getitem__(self, idx):
            # Extract sequence of stock data for the given index
            data_seq = self.data[self.feature_columns].iloc[idx:idx + self.sequence_length].values
            symbol = self.data['Symbol'].iloc[idx + self.sequence_length - 1]  # Symbol for the current sequence

            # Convert to torch tensors
            data_seq = torch.tensor(data_seq, dtype=torch.float32)

            return data_seq, symbol



        # Model instantiation
        input_size = len(dataset.feature_columns)
        hidden_size = 64  # Number of LSTM units
        num_layers = 2  # Number of LSTM layers
        output_size = 128  # The desired embedding size

        model = LSTMNetwork(input_size, hidden_size, num_layers, output_size)

        # Example forward pass
        for batch_data, _ in dataloader:
            embeddings = model(batch_data)

    # Instantiate the dataset and dataloader
    sequence_length = 30  # You can choose this based on how long the LSTM input sequence should be
    # csv_file = "C:\Users\Ashitosh\Desktop\capstone\hats\dataset"

    dataset = StockDataset(csv_file=csv_file, sequence_length=sequence_length)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)

    # Example: Iterate through the dataloader
    for batch_data, symbols in dataloader:
        print(batch_data.shape)  # Expected shape: [batch_size, sequence_length, num_features]
        print(symbols)  # Stock symbols

    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    model_name = config.model_type
    exp_name = '%s_%s_%s_%s_%s_%s_%s_%s' % (config.data_type, model_name,
                                            str(config.test_phase), str(config.test_size),
                                            str(config.train_proportion), str(config.lr),
                                            str(config.dropout), str(config.lookback))
    if not (os.path.exists(os.path.join(config.save_dir, exp_name))):
        os.makedirs(os.path.join(config.save_dir, exp_name))

    sess = tf.Session(config=run_config)
    model = init_prediction_model(config)
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init)

    def model_summary(logger, slim=None):
        model_vars = tf.trainable_variables()
        slim.model_analyzer.analyze_vars(model_vars, print_info=True)

    model_summary(logger)

    # Simulate long-running training process
    simulate_long_running_process(duration_in_hours=3)  # Run for at least 3 hours

    # Training
    evaluator = Evaluator(config, logger)
    trainer = Trainer(sess, model, dataset, config, logger, evaluator)
    trainer.train()

    # Testing
    loader = tf.train.Saver(max_to_keep=None)
    loader.restore(sess, tf.train.latest_checkpoint(os.path.join(config.save_dir, exp_name)))
    print("Loaded best evaluation model.")

    test_loss, report_all, report_topk = evaluator.evaluate(sess, model, dataset, 'test', trainer.best_f1['neighbors'])
    te_pred_rate, te_acc, te_cpt_acc, te_mac_f1, te_mic_f1, te_exp_rt = report_all
    logstr = 'EPOCH {} TEST ALL \nloss : {:2.4f} accuracy : {:2.4f} hit ratio : {:2.4f} pred_rate : {} macro f1 : {:2.4f} micro f1 : {:2.4f} expected return : {:2.4f}' \
        .format(trainer.best_f1['epoch'], test_loss, te_acc, te_cpt_acc, te_pred_rate, te_mac_f1, te_mic_f1, te_exp_rt)
    logger.info(logstr)

    te_pred_rate, te_acc, te_cpt_acc, te_mac_f1, te_mic_f1, te_exp_rt = report_topk
    logstr = 'EPOCH {} TEST TopK \nloss : {:2.4f} accuracy : {:2.4f} hit ratio : {:2.4f} pred_rate : {} macro f1 : {:2.4f} micro f1 : {:2.4f} expected return : {:2.4f}' \
        .format(trainer.best_f1['epoch'], test_loss, te_acc, te_cpt_acc, te_pred_rate, te_mac_f1, te_mic_f1, te_exp_rt)
    logger.info(logstr)

    # Print Log
    with open('%s_log.log' % model_name, 'a') as out_:
        out_.write("%d phase\n" % (config.test_phase))
        out_.write("%f\t%f\t%f\t%f\t%f\t%s\t%f\t%f\t%f\t%f\t%f\t%s\t%d\n" % (
            report_all[1], report_all[2], report_all[3], report_all[4], report_all[5], str(report_all[0]),
            report_topk[1], report_topk[2], report_topk[3], report_topk[4], report_topk[5], str(report_topk[0]),
            trainer.best_f1['epoch']))

        class LSTMNetwork(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, output_size):
                super(LSTMNetwork, self).__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                self.fc = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                lstm_out = lstm_out[:, -1, :]  # Get the output of the last time step
                output = self.fc(lstm_out)
                return output


if __name__ == '__main__':
    randomize(3)
    main()

# import os, time, argparse
# import numpy as np
# import pandas as pd
# import tensorflow as tf
# import tensorflow.contrib.slim as slim
#
# from logger import set_logger
# from config import get_args
# from dataset import StockDataset
# from trainer import Trainer
# from evaluator import Evaluator
# from models.HATS import HATS
#
# def init_prediction_model(config):
#     with tf.variable_scope("model"):
#         if config.model_type == "HATS":
#             model = HATS(config)
#     return model
#
# def main():
#     config = get_args()
#     logger = set_logger(config)
#     dataset = StockDataset(config)
#     config.num_relations = dataset.num_relations
#     config.num_companies = dataset.num_companies
#
#     run_config = tf.ConfigProto()
#     run_config.gpu_options.allow_growth = True
#     model_name = config.model_type
#     exp_name = '%s_%s_%s_%s_%s_%s_%s_%s'%(config.data_type, model_name,
#                                         str(config.test_phase), str(config.test_size),
#                                         str(config.train_proportion), str(config.lr),
#                                         str(config.dropout), str(config.lookback))
#     if not (os.path.exists(os.path.join(config.save_dir, exp_name))):
#         os.makedirs(os.path.join(config.save_dir, exp_name))
#
#     sess = tf.Session(config=run_config)
#     model = init_prediction_model(config)
#     init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
#     sess.run(init)
#
#     def model_summary(logger):
#         model_vars = tf.trainable_variables()
#         slim.model_analyzer.analyze_vars(model_vars, print_info=True)
#     model_summary(logger)
#
#     #Training
#     evaluator = Evaluator(config, logger)
#     trainer = Trainer(sess, model, dataset, config, logger, evaluator)
#     trainer.train()
#
#     #Testing
#     loader = tf.train.Saver(max_to_keep=None)
#     loader.restore(sess, tf.train.latest_checkpoint(os.path.join(config.save_dir, exp_name)))
#     print("load best evaluation model")
#
#     test_loss, report_all, report_topk = evaluator.evaluate(sess, model, dataset, 'test', trainer.best_f1['neighbors'])
#     te_pred_rate, te_acc, te_cpt_acc, te_mac_f1, te_mic_f1, te_exp_rt = report_all
#     logstr = 'EPOCH {} TEST ALL \nloss : {:2.4f} accuracy : {:2.4f} hit ratio : {:2.4f} pred_rate : {} macro f1 : {:2.4f} micro f1 : {:2.4f} expected return : {:2.4f}'\
#             .format(trainer.best_f1['epoch'],test_loss,te_acc,te_cpt_acc,te_pred_rate,te_mac_f1,te_mic_f1,te_exp_rt)
#     logger.info(logstr)
#
#     te_pred_rate, te_acc, te_cpt_acc, te_mac_f1, te_mic_f1, te_exp_rt = report_topk
#     logstr = 'EPOCH {} TEST TopK \nloss : {:2.4f} accuracy : {:2.4f} hit ratio : {:2.4f} pred_rate : {} macro f1 : {:2.4f} micro f1 : {:2.4f} expected return : {:2.4f}'\
#             .format(trainer.best_f1['epoch'],test_loss,te_acc,te_cpt_acc,te_pred_rate,te_mac_f1,te_mic_f1,te_exp_rt)
#     logger.info(logstr)
#
#     #Print Log
#     with open('%s_log.log'%model_name, 'a') as out_:
#         out_.write("%d phase\n"%(config.test_phase))
#         out_.write("%f\t%f\t%f\t%f\t%f\t%s\t%f\t%f\t%f\t%f\t%f\t%s\t%d\n"%(
#             report_all[1], report_all[2], report_all[3], report_all[4], report_all[5], str(report_all[0]),
#             report_topk[1], report_topk[2], report_topk[3], report_topk[4], report_topk[5], str(report_topk[0]),
#             trainer.best_f1['epoch']))
#
# if __name__ == '__main__':
#     main()




