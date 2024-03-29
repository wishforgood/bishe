##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import math
import logging


class LR_Scheduler(object):
    """Learning Rate Scheduler

    Step mode: ``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``

    Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``

    Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ 0.9``

    Args:
        args:  :attr:`args.lr_scheduler` lr scheduler mode (`cos`, `poly`),
          :attr:`args.lr` base learning rate, :attr:`args.epochs` number of epochs,
          :attr:`args.lr_step`

        iters_per_epoch: number of iterations per epoch
    """

    def __init__(self, mode, base_lr, num_epochs, iters_per_epoch=0, freezn=False,
                 lr_step=0, decode_lr_factor=1, aspp=None, warmup_epochs=0, logger=logging):
        self.mode = mode
        self.freezn = freezn
        self.aspp = aspp
        self.decode_lr_factor = decode_lr_factor
        print('Using {} LR Scheduler!'.format(self.mode))
        self.lr = base_lr
        if mode == 'step':
            assert lr_step
        self.lr_step = lr_step
        self.iters_per_epoch = iters_per_epoch
        self.N = num_epochs * iters_per_epoch
        self.epoch = -1
        self.warmup_iters = warmup_epochs * iters_per_epoch
        self.logger = logger
        self.logger.info('Using {} LR Scheduler!'.format(self.mode))

    def __call__(self, optimizer, i, epoch, best_pred):
        T = epoch * self.iters_per_epoch + i
        if self.mode == 'cos':
            lr = 0.5 * self.lr * (1 + math.cos(1.0 * T / self.N * math.pi))
        elif self.mode == 'poly':
            lr = self.lr * pow((1 - 1.0 * T / self.N), 0.9)
        elif self.mode == 'step':
            lr = self.lr * (0.1 ** (epoch // self.lr_step))
        elif self.mode == 'simple':
            if (epoch + 1) % 30 == 0:
                self.lr = self.lr / 2
            else:
                self.lr = self.lr
            lr = self.lr
        elif self.mode == 'none':
            lr = self.lr
        else:
            raise NotImplemented
        # warm up lr schedule
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = lr * 1.0 * T / self.warmup_iters
        if epoch > self.epoch:
            print('\n=>Epoches %i, learning rate = %.6f, \
                previous best = %.4f' % (epoch, lr, best_pred))
            self.epoch = epoch
        assert lr >= 0
        self._adjust_learning_rate(optimizer, lr)

    def _adjust_learning_rate(self, optimizer, lr):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:
            if self.freezn:
                for i in range(0, len(optimizer.param_groups)):
                    optimizer.param_groups[i]['lr'] = lr
            else:
                # enlarge the lr at the head
                optimizer.param_groups[0]['lr'] = lr
                if self.aspp is not None:
                    optimizer.param_groups[1]['lr'] = lr
                    for i in range(2, len(optimizer.param_groups)):
                        optimizer.param_groups[i]['lr'] = lr * self.decode_lr_factor
                else:
                    for i in range(1, len(optimizer.param_groups)):
                        optimizer.param_groups[i]['lr'] = lr * self.decode_lr_factor
                # optimizer.param_groups[2]['lr'] = lr * 10