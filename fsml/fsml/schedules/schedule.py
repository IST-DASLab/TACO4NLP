__all__ = [
    'SparsitySchedule'
]


class SparsitySchedule:

    inter_funcs = ('linear', 'cubic')

    def __init__(
        self, 
        init_sparsity, 
        final_sparsity, 
        total_steps, 
        cooldown_fraction = 0.0, 
        inter_func = 'linear'
    ):
        assert inter_func in self.inter_funcs
        assert 0.0 <= cooldown_fraction <= 1.0
        self.init_sparsity = init_sparsity
        self.final_sparsity = final_sparsity
        self.cooldown_fraction = cooldown_fraction
        self.total_steps = total_steps
        self.inter_func = inter_func

    def get_sparsity_for_step(self, step):
        last_pruning_step = self.total_steps * (1 - self.cooldown_fraction)
        rel_progress = min(step / last_pruning_step, 1.0)
        if self.inter_func == 'linear':
            alpha = rel_progress
        elif self.inter_func == 'cubic':
            alpha = 1 - (1 - rel_progress) ** 3
        return alpha * (self.final_sparsity - self.init_sparsity) + self.init_sparsity
