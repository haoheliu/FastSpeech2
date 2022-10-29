import torch
import numpy as np


class ScheduledOptim:
    """ A simple wrapper class for learning rate scheduling """

    def __init__(self, model, train_config, model_config, current_step):
        self.init_lr = train_config["optimizer"]["lr"]
        self._optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.init_lr,
            betas=train_config["optimizer"]["betas"],
            eps=train_config["optimizer"]["eps"],
            weight_decay=train_config["optimizer"]["weight_decay"],
        )
        self.n_warmup_steps = train_config["optimizer"]["warm_up_step"]
        self.anneal_steps = train_config["optimizer"]["anneal_steps"]
        self.anneal_rate = train_config["optimizer"]["anneal_rate"]
        self.current_step = current_step

    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()

    def zero_grad(self):
        # print(self.init_lr)
        self._optimizer.zero_grad()

    def load_state_dict(self, path):
        self._optimizer.load_state_dict(path)

    def _get_lr_scale(self, lr):
        init_lr = self.init_lr
        for s in self.anneal_steps:
            if self.current_step > s:
                init_lr = init_lr * self.anneal_rate
        return init_lr

    def _update_learning_rate(self):
        """ Learning rate scheduling per step """
        self.current_step += 1

        for param_group in self._optimizer.param_groups:
            lr = self._get_lr_scale(param_group["lr"])
            param_group["lr"] = lr
    
    def get_lr(self):
        lr = []
        for param_group in self._optimizer.param_groups:
            lr.append(param_group["lr"])
        return lr