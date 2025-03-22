# utils/tuning.py

from skopt import gp_minimize
from ray import tune
from ray.tune.schedulers import ASHAScheduler

class SkoptTuner:
    def __init__(self, model_class, metric_func, space):
        self.model_class = model_class
        self.metric_func = metric_func
        self.space = space

    def optimize(self, X, y_dummy):
        def objective(params):
            model = self.model_class(*params)
            preds = model.fit_predict(X)
            y_pred = (preds == -1).astype(int)
            metrics = self.metric_func(y_dummy, y_pred)
            return -metrics['F1']  # minimize negative F1

        result = gp_minimize(objective, self.space, n_calls=20, random_state=42)
        return result.x


class RayTuner:
    def __init__(self, train_func, param_space, num_samples=20, max_epochs=10):
        self.train_func = train_func
        self.param_space = param_space
        self.num_samples = num_samples
        self.max_epochs = max_epochs

    def optimize(self):
        # ASHA Scheduler for production
        scheduler = ASHAScheduler(
            metric="F1",  # Can also switch to "val_loss" or "accuracy" if needed
            mode="max",

            # For DEV environment:
            # grace_period=1,         # minimum epochs before stopping
            # reduction_factor=2      # aggressiveness of pruning

            # For Runpod environment:
            grace_period=3,         # minimum epochs before stopping
            reduction_factor=2      # aggressiveness of pruning
        )

        analysis = tune.run(
            self.train_func,
            config=self.param_space,
            num_samples=self.num_samples,
            stop={"training_iteration": self.max_epochs},

            # === LOCAL DEV MODE ===
            # resources_per_trial={"cpu": 4, "gpu": 1},
            # max_concurrent_trials=1,

            # === RUNPOD / PRODUCTION MODE ===
            resources_per_trial={"cpu": 8, "gpu": 1},
            max_concurrent_trials=4,  # Customize based on your Runpod instance
            scheduler=scheduler
        )

        return analysis.best_config
