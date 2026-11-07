from copy import deepcopy
import numpy as np
from . import optim


class Solver:
    def __init__(self, model, data, *, update_rule='sgd', optim_config=None,
                 lr_decay=1.0, batch_size=100, num_epochs=10, print_every=10,
                 verbose=True):
        self.model = model
        self.X_train, self.y_train = data['X_train'], data['y_train']
        self.X_val, self.y_val = data['X_val'], data['y_val']
        self.update_rule = getattr(optim, update_rule)
        self.optim_configs = {k: deepcopy(optim_config or {}) for k in model.params}
        self.lr_decay, self.batch_size, self.num_epochs = lr_decay, batch_size, num_epochs
        self.print_every, self.verbose = print_every, verbose
        self.loss_history, self.train_acc_history, self.val_acc_history = [], [], []
        self.best_val_acc = -1.0

    def check_accuracy(self, X, y, num_samples=None, batch_size=128):
        if num_samples is not None and len(X) > num_samples:
            idx = np.random.choice(len(X), num_samples, replace=False)
            X, y = X[idx], y[idx]
        correct = 0
        for start in range(0, len(y), batch_size):
            end = start + batch_size
            correct += (self.model.loss(X[start:end]).argmax(axis=1) == y[start:end]).sum()
        return correct / len(y)

    def train(self):
        iteration = 0
        for epoch in range(self.num_epochs):
            order = np.random.permutation(len(self.y_train))
            for start in range(0, len(order), self.batch_size):
                idx = order[start:start+self.batch_size]
                loss, grads = self.model.loss(self.X_train[idx], self.y_train[idx])
                self.loss_history.append(loss)
                if self.verbose and iteration % self.print_every == 0:
                    print(f'(Iteration {iteration}) loss: {loss:.6f}')
                for name, w in self.model.params.items():
                    self.model.params[name], self.optim_configs[name] = self.update_rule(
                        w, grads[name], self.optim_configs[name])
                iteration += 1
            train_acc = self.check_accuracy(self.X_train, self.y_train)
            val_acc = self.check_accuracy(self.X_val, self.y_val)
            self.train_acc_history.append(train_acc)
            self.val_acc_history.append(val_acc)
            if self.verbose:
                print(f'(Epoch {epoch+1}/{self.num_epochs}) train acc: {train_acc:.4f}, val acc: {val_acc:.4f}')
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_params = deepcopy(self.model.params)
                self.best_bn_params = deepcopy(getattr(self.model, 'bn_params', None))
            for config in self.optim_configs.values():
                config['learning_rate'] *= self.lr_decay
        self.model.params = self.best_params
        if self.best_bn_params is not None:
            self.model.bn_params = self.best_bn_params
