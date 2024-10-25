import tensorflow as tf
import pdb, time

class BaseTrain:
    def __init__(self, sess, model, data, config, logger, evaluator):
        self.model = model
        self.logger = logger
        self.config = config
        self.sess = sess
        self.data = data
        self.evaluator = evaluator
        self.best_f1 = {'topk':dict(), 'all':{'macroF1':0}}

    def train(self):
        te_loss_hist, te_acc_hist, te_acc_k_hist = [], [], []
        prev = -1
        for cur_epoch in range(self.model.cur_epoch_tensor.eval(self.sess), self.config.n_epochs + 1, 1):
            loss, report_all, report_topk = self.train_epoch()
            self.sess.run(self.model.increment_cur_epoch_tensor)
            if cur_epoch % self.config.print_step == 0:
                pred_rate, acc, cpt_acc, mac_f1, mic_f1, exp_rt = report_all
                logstr = 'EPOCH {} TRAIN ALL \nloss : {:2.4f} accuracy : {:2.4f} hit ratio : {:2.4f} pred_rate : {} macro f1 : {:2.4f} micro f1 : {:2.4f} expected return : {:2.4f}'\
                        .format(cur_epoch+1,loss,acc,cpt_acc,pred_rate,mac_f1,mic_f1,exp_rt)
                self.logger.info(logstr)
                pred_rate, acc, cpt_acc, mac_f1, mic_f1, exp_rt = report_topk
                logstr = 'EPOCH {} TRAIN TOPK \nloss : {:2.4f} accuracy : {:2.4f} hit ratio : {:2.4f} pred_rate : {} macro f1 : {:2.4f} micro f1 : {:2.4f} expected return : {:2.4f}'\
                        .format(cur_epoch+1,loss,acc,cpt_acc,pred_rate,mac_f1, mic_f1,exp_rt)
                self.logger.info(logstr)
            if cur_epoch % self.config.eval_step == 0:
                if self.config.model_type in ['graph-cnn','mlp', 'lstm', 'cnn']:
                    neighbors = None
                else:
                    neighbors = self.sample_neighbors()
                te_loss, report_all, report_topk = self.evaluator.evaluate(self.sess, self.model, self.data, 'eval', neighbors)
                te_pred_rate, te_acc, te_cpt_acc, te_mac_f1, te_mic_f1, te_exp_rt = report_all

                logstr = '\n\n EPOCH {} EVAL ALL \nloss : {:2.4f} accuracy : {:2.4f} hit ratio : {:2.4f} pred_rate : {} macro f1 : {:2.4f} micro f1 : {:2.4f} expected return : {:2.4f}'\
                        .format(cur_epoch+1,te_loss,te_acc,te_cpt_acc,te_pred_rate,te_mac_f1, te_mic_f1,te_exp_rt)
                self.logger.info(logstr)
                te_loss_hist.append(te_loss)
                te_acc_hist.append(te_acc)

                if te_mac_f1 > prev:
                    #if cur_epoch < 10:
                    #    continue
                    prev = te_mac_f1
                    self.model.save(self.sess)
                    time.sleep(5)
                    self.best_f1['topk']['acc'] = report_topk[1]
                    self.best_f1['topk']['hit'] = report_topk[2]
                    self.best_f1['topk']['macroF1'] = report_topk[3]
                    self.best_f1['topk']['microF1'] = report_topk[4]
                    self.best_f1['topk']['return'] = report_topk[5]
                    self.best_f1['topk']['ratio'] = report_topk[0]

                    self.best_f1['all']['acc'] = report_all[1]
                    self.best_f1['all']['hit'] = report_all[2]
                    self.best_f1['all']['macroF1'] = report_all[3]
                    self.best_f1['all']['microF1'] = report_all[4]
                    self.best_f1['all']['return'] = report_all[5]
                    self.best_f1['all']['ratio'] = report_all[0]

                    self.best_f1['epoch'] = cur_epoch
                    self.best_f1['lookback'] = self.config.lookback
                    self.best_f1['neighbors'] = neighbors

                te_pred_rate, te_acc, te_cpt_acc, te_f1, te_mcc, te_exp_rt = report_topk
                logstr = 'EPOCH {} EVAL TOPK \nloss : {:2.4f} accuracy : {:2.4f} hit ratio : {:2.4f} pred_rate : {} macro f1 : {:2.4f} micro f1 : {:2.4f} expected return : {:2.4f} \n\n'\
                        .format(cur_epoch+1,te_loss,te_acc,te_cpt_acc,te_pred_rate,te_mac_f1, te_mic_f1,te_exp_rt)
                self.logger.info(logstr)
                te_acc_k_hist.append(te_acc)

        self.logger.info(te_loss_hist)
        self.logger.info(te_acc_hist)


    def train_epoch(self):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def train_step(self):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError
