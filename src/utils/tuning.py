# utils/tuning.py
# checked

from itertools import product
import logging
from skopt import gp_minimize
import traceback
import ray
from ray import tune
from ray.tune.schedulers import ASHAScheduler

def manual_gru_search(train_func, param_grid):
    # Check if param_grid is a list (manual config mode)
    if isinstance(param_grid, list):
        best_score = float("-inf")
        best_config = None
        for config in param_grid:
            logging.info(f"🔍 Trying manual config: {config}")
            print(f"\n🚀 Trying manual config: {config}")
            try:
                score = train_func(config)
                print(f"✅ Config {config} got score {score}")
                if score > best_score:
                    best_score = score
                    best_config = config
            except Exception as e:
                print(f"❌ Error with config {config}: {e}")
                traceback.print_exc()
                continue
        return best_config

    # Default grid search mode (dict of hyperparam lists)
    keys, values = zip(*param_grid.items())
    best_score = float("-inf")
    best_config = None
    for v in product(*values):
        config = dict(zip(keys, v))
        logging.info(f"🔍 Trying grid config: {config}")
        print(f"\n🚀 Trying grid config: {config}")
        try:
            score = train_func(config)
            print(f"✅ Config {config} got score {score}")
            if score > best_score:
                best_score = score
                best_config = config
        except Exception as e:
            print(f"❌ Error with config {config}: {e}")
            continue
    return best_config


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
        # ✅ Manual Ray init (for pods or Jupyter)
        if not ray.is_initialized():
            ray.init(
                num_cpus=2,
                num_gpus=1,
                include_dashboard=False,
                _temp_dir="/tmp/ray",
                ignore_reinit_error=True
            )


        # ASHA scheduler (aggressive early stopping)
        scheduler = ASHAScheduler(
            metric="F1",  # You can change this to val_loss or accuracy
            mode="max",
            grace_period=3,
            reduction_factor=2
        )

        analysis = tune.run(
            self.train_func,
            config=self.param_space,
            num_samples=self.num_samples,
            stop={"training_iteration": self.max_epochs},
            resources_per_trial={"cpu": 2, "gpu": 1},
            max_concurrent_trials=2,
            scheduler=scheduler
        )

        return analysis.best_config
