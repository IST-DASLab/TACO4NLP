__all__ = [
    'CyclicLinearLR'
]

class CyclicLinearLR:

    def __init__(
        self, 
        optimizer, 
        cycle_steps, 
        start_factor=1.0, 
        end_factor=0.0, 
        num_warmup_steps=0
    ) -> None:
        self.optimizer = optimizer
        self.cycle_steps = cycle_steps
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.num_warmup_steps = num_warmup_steps
        self.total_iters = 0
        # backup base learning rate
        self.base_lrs = [group['lr'] for group in self.optimizer.param_groups]

    def step(self):
        adj_rel_step = (self.total_iters % self.cycle_steps) / self.cycle_steps
        # get current mult factor
        factor = self.start_factor + adj_rel_step * (self.end_factor - self.start_factor)
        # multiply by warmup factor if needed
        if self.num_warmup_steps and self.total_iters < self.num_warmup_steps:
            factor *= (self.total_iters / self.num_warmup_steps)
        for group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            group['lr'] = factor * base_lr
        # update current step
        self.total_iters += 1
        