import numpy as np


class CLR(object):
    """
    The method is described in paper : https://arxiv.org/abs/1506.01186 to find
    out optimum learning rate. The learning rate is increased from lower value
    to higher per iteration for some iterations till loss starts exploding. The
    learning rate one power lower than the one where loss is minimum is chosen
    as optimum learning rate for training.
    Args:
        opt     Optimizer used in training.
        itr     Total number of iterations used for this test run.
                The learning rate increasing factor is calculated based on this
                iteration number.
        base_lr The lower boundary for learning rate which will be used as
                initial learning rate during test run. It is adviced to start
                from small learning rate value like 1e-4.
                Default value is 1e-5
        max_lr  The upper boundary for learning rate. This value defines
                amplitude for learning rate increase(max_lr-base_lr). max_lr
                value may not be reached in test run as loss may explode before
                reaching max_lr. It is adviced to use 10*base_lr.
    """

    def __init__(self, base_lr, max_lr, mode, step_size=2000, gamma=0.99994):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.gamma = gamma
        self.mode = mode
        self.itr = 0

    def calc(self):
        self.itr += 1
        lr = self.calc_lr()
        return np.float(lr), 1

    def calc_lr(self):
        cycle = np.floor(1 + self.itr/(2*self.step_size))
        x = np.abs(self.itr/self.step_size - 2*cycle + 1)
        lr = self.base_lr + (self.max_lr - self.base_lr)*max(0, 1-x)
        if self.mode == 'triangular2':
            lr /= float(2**(cycle-1))
        elif self.mode == 'exp_range':
            lr *= self.gamma**self.itr
        return lr


class OneCycle(object):
    """
    In paper (https://arxiv.org/pdf/1803.09820.pdf), author suggests to do
    one cycle during whole run with 2 steps of equal length. During first step,
    increase the learning rate from lower learning rate to higher learning rate.
    And in second step, decrease it from higher to lower learning rate. This is
    Cyclic learning rate policy. Author suggests one addition to this. - During
    last few hundred/thousand iterations of cycle reduce the learning rate to
    1/100th or 1/1000th of the lower learning rate.
    Also, Author suggests that reducing momentum when learning rate is
    increasing. So, we make one cycle of momentum also with learning rate -
    Decrease momentum when learning rate is increasing and increase momentum
    when learning rate is decreasing.
    Args:
        total           Total number of iterations including all epochs
        max_lr          The optimum learning rate. This learning rate will be
                        used as highest learning rate. The learning rate will
                        fluctuate between max_lr to max_lr/div and then
                        (max_lr/div)/div.
        momentum_vals   The maximum and minimum momentum values between which
                        momentum will fluctuate during cycle.
                        Default values are (0.95, 0.85)
        prcnt           The percentage of cycle length for which we annihilate
                        learning rate way below the lower learnig rate.
                        The default value is 10
        div             The division factor used to get lower boundary of
                        learning rate. This will be used with max_lr value to
                        decide lower learning rate boundary. This value is also
                        used to decide how much we annihilate the learning
                        rate below lower learning rate.
                        The default value is 10.
    """

    def __init__(self, total, max_lr, momentum_vals=(0.95, 0.85), prct=10,
                 div=10):
        self.total = total
        self.max_lr = max_lr
        self.high_mom = momentum_vals[0]
        self.low_mom = momentum_vals[1]
        self.div = div
        self.itr = 0
        self.step_len = int(self.total * (1 - prct / 100) / 2)

    def calc(self):
        self.itr += 1
        lr = self.calc_lr()
        mom = self.calc_mom()
        return np.float(lr), np.float(mom)

    def calc_lr(self):
        if self.itr == self.total:
            return self.max_lr / self.div
        if self.itr > 2 * self.step_len:
            ratio = (self.itr - 2 * self.step_len) / (
                    self.total - 2 * self.step_len)
            lr = self.max_lr * (1 - 0.99 * ratio) / self.div
        elif self.itr > self.step_len:
            ratio = 1 - (self.itr - self.step_len) / self.step_len
            lr = self.max_lr * (1 + ratio * (self.div - 1)) / self.div
        else:
            ratio = self.itr / self.step_len
            lr = self.max_lr * (1 + ratio * (self.div - 1)) / self.div
        return lr

    def calc_mom(self):
        if self.itr == self.total:
            self.itr = 0
            return self.high_mom
        if self.itr > 2 * self.step_len:
            mom = self.high_mom
        elif self.itr > self.step_len:
            ratio = (self.itr - self.step_len) / self.step_len
            mom = self.low_mom + ratio * (self.high_mom - self.low_mom)
        else:
            ratio = self.itr / self.step_len
            mom = self.high_mom - ratio * (self.high_mom - self.low_mom)
        return mom

