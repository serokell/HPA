import matplotlib.pyplot as plt
import math


class LRFinder:
    def __init__(
        self, optimizer, 
        min_lr=1e-4, max_lr=2e-2, 
        steps_per_epoch=None, epochs=None
    ):

        self.optimizer = optimizer
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.total_iterations = steps_per_epoch * epochs
        self.iteration = 0
        self.history = {}

        self.batch_step(self.iteration)

    def get_lr(self):
        '''Calculate the learning rate.'''
        x = (self.iteration % self.total_iterations) / self.total_iterations
        lr = self.min_lr + (self.max_lr - self.min_lr) * x

        lrs = list()
        for param_group in self.optimizer.param_groups:
            lrs.append(lr)
        return lrs

    def batch_step(self, batch_iteration=None, logs=None):
        self.iteration = batch_iteration or self.iteration + 1
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

        if logs is not None:
            self.history.setdefault('lr', []).append(lr)
            self.history.setdefault('iterations', []).append(self.iteration)

            for k, v in logs.items():
                self.history.setdefault(k, []).append(v)
 
    def plot_lr(self):
        '''Helper function to quickly inspect the learning rate schedule.'''
        plt.plot(self.history['iterations'], self.history['lr'])
        plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Learning rate')
        
    def plot_loss(self):
        '''Helper function to quickly observe the learning rate experiment results.'''
        plt.plot(self.history['lr'], self.history['loss'])
        plt.xscale('log')
        plt.xlabel('Learning rate')
        plt.ylabel('Loss')


def find_lr(model, datagen_params, min_lr=1e-5, max_lr=1e-2, epochs=3):
    train_datagen, val_datagen = get_datagens(max_negatives=2000, **datagen_params)
    opt = torch.optim.SGD(model.parameters(), lr=min_lr, momentum=.9, weight_decay=1e-4)
    learner = RetinaLearner(model=model, opt=opt, loss=None, clf_loss=None, metrics=[], clf_reg_alpha=.5, ignored_keys=['clf_out'])

    print('steps per epoch: {}'.format(len(train_datagen)))
    lr_scheduler = LRFinder(learner.opt, min_lr=min_lr, max_lr=max_lr, steps_per_epoch=len(train_datagen), epochs=epochs)
    learner, history = orchestrate(
        learner=learner, train_datagen=train_datagen, val_datagen=val_datagen, epochs=epochs,
        lr_scheduler=lr_scheduler, checkpoints_pth=None, nb_freezed_epchs=-1, df=datagen_params['df'],
    )
    return learner, lr_scheduler


class Pilo:
    def __init__(
        self, optimizer, 
        min_lr=1e-4, max_lr=2e-2, 
        coeff = 1.,
        steps_per_epoch=None
    ):

        self.optimizer = optimizer
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.total_iterations = steps_per_epoch
        self.iteration = 0
        self.history = {}
        self.coeff = coeff
        self.batch_step(self.iteration)

    def get_lr(self):
        '''Calculate the learning rate.'''
        x = float(self.iteration % self.total_iterations) / self.total_iterations
        lr = self.max_lr - (self.max_lr - self.min_lr) * x

        lrs = list()
        for param_group in self.optimizer.param_groups:
            lrs.append(lr)
        return lrs

    def batch_step(self, batch_iteration=None, logs=None):
        self.iteration = batch_iteration or self.iteration + 1
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

        if logs is not None:
            self.history.setdefault('lr', []).append(lr)
            self.history.setdefault('iterations', []).append(self.iteration)

    def step(self, batch_iteration=None, logs=None):
        self.max_lr *= self.coeff
        self.min_lr *= self.coeff
 
    def plot_lr(self):
        '''Helper function to quickly inspect the learning rate schedule.'''
        plt.plot(self.history['lr'])
#         plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Learning rate')


class PiloExt:
    def __init__(
        self, optimizer, 
        multiplier=.1,
        coeff=1.,
        steps_per_epoch=None
    ):

        self.optimizer = optimizer
        self.multiplier = multiplier
        self.total_iterations = steps_per_epoch

        self.param_groups_old = list()
        for param_group in self.optimizer.param_groups:
            self.param_groups_old.append(float(param_group['lr']))

        self.iteration = 0
        self.history = {}
        self.coeff = coeff
        self.batch_step(self.iteration)
        
    def get_lr(self):
        '''Calculate the learning rate.'''
        x = float(self.iteration % self.total_iterations) / self.total_iterations

        lrs = list()
        for i, param_group in enumerate(self.optimizer.param_groups):
            lrs.append(self.param_groups_old[i] * (1 - x) + self.param_groups_old[i] * self.multiplier * x)
        return lrs

    def batch_step(self, batch_iteration=None, logs=None):
        self.iteration = batch_iteration or self.iteration + 1
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

        if logs is not None:
            self.history.setdefault('lr', []).append(lr)
            self.history.setdefault('iterations', []).append(self.iteration)

    def step(self, batch_iteration=None, logs=None):
        for i, param_group in enumerate(self.param_groups_old):
            self.param_groups_old[i] *= self.coeff
 
    def plot_lr(self):
        '''Helper function to quickly inspect the learning rate schedule.'''
        plt.plot(self.history['lr'])
#         plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Learning rate')


class CosinePiloExt(PiloExt):
    def get_lr(self):
        '''Calculate the learning rate.'''
        x = float(self.iteration % self.total_iterations) / self.total_iterations

        lrs = list()
        for i, param_group in enumerate(self.optimizer.param_groups):
            lrs.append(
                self.param_groups_old[i] * self.multiplier 
                + (self.param_groups_old[i] - self.param_groups_old[i] * self.multiplier) 
                * (1 + math.cos(math.pi * x)) / 2
            )
        return lrs
