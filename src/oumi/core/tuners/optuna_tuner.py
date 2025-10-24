# Copyright 2025 - Oumi
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from typing import Any, Callable, Union

import optuna

from oumi.core.configs import TuningConfig
from oumi.core.configs.params.tuning_params import ParamType, TuningParams
from oumi.core.tuners.base_tuner import BaseTuner


class OptunaTuner(BaseTuner):
    """Optuna-based hyperparameter tuner implementation."""

    def __init__(self, tuning_params: TuningParams):
        """Initializes the Optuna based hyperparameter tuner.

        Args:
            tuning_params (TuningParams): _description_
        """
        super().__init__(tuning_params)
        self._study: optuna.Study

    def create_study(self) -> None:
        """Create an Optuna study with multi-objective optimization support."""
        # Determine optimization directions
        directions = []
        for direction in self.tuning_params.evaluation_direction:
            if direction == "minimize":
                directions.append(optuna.study.StudyDirection.MINIMIZE)
            else:
                directions.append(optuna.study.StudyDirection.MAXIMIZE)

        # Create study
        self._study = optuna.create_study(
            study_name=self.tuning_params.tuning_study_name,
            directions=directions,
            storage=None,  # Can be configured for persistence
        )

    def suggest_parameters(self, trial: optuna.Trial) -> dict[str, Any]:
        """Suggest parameters using Optuna's suggest methods."""
        suggested_params = {}

        for (
            param_name,
            param_spec,
        ) in self.tuning_params.tunable_training_params.items():
            if isinstance(param_spec, list):
                # Categorical parameter
                suggested_params[param_name] = trial.suggest_categorical(
                    param_name, param_spec
                )
            elif isinstance(param_spec, dict):
                param_type = ParamType(param_spec["type"])

                if param_type == ParamType.CATEGORICAL:
                    suggested_params[param_name] = trial.suggest_categorical(
                        param_name, param_spec["choices"]
                    )
                elif param_type == ParamType.INT:
                    suggested_params[param_name] = trial.suggest_int(
                        param_name,
                        param_spec["low"],
                        param_spec["high"],
                    )
                elif param_type == ParamType.FLOAT:
                    suggested_params[param_name] = trial.suggest_float(
                        param_name,
                        param_spec["low"],
                        param_spec["high"],
                    )
                elif param_type == ParamType.LOGUNIFORM:
                    suggested_params[param_name] = trial.suggest_loguniform(
                        param_name,
                        param_spec["low"],
                        param_spec["high"],
                    )
                elif param_type == ParamType.UNIFORM:
                    suggested_params[param_name] = trial.suggest_uniform(
                        param_name,
                        param_spec["low"],
                        param_spec["high"],
                    )
                else:
                    raise ValueError(f"Unsupported parameter type: {param_type}")
            else:
                raise ValueError(
                    f"Parameter specification for {param_name} is invalid."
                )
        return suggested_params

    def optimize(
        self,
        objective_fn: Callable[[dict[str, Any], dict[str, Any], int], dict[str, float]],
        n_trials: int,
    ) -> dict[str, Any]:
        """Run Optuna optimization."""
        if self._study is None:
            self.create_study()

        def _objective(trial: optuna.Trial) -> Union[float, list[float]]:
            # Get suggested parameters
            params = self.suggest_parameters(trial)

            # Run objective function (training + evaluation)
            metrics = objective_fn(
                params,
                self.tuning_params.fixed_training_params,
                trial.number,
            )

            # Return metric values in the order specified
            metric_values = [
                metrics[metric_name]
                for metric_name in self.tuning_params.evaluation_metrics
            ]

            # Return single value or list for multi-objective
            return metric_values[0] if len(metric_values) == 1 else metric_values

        # Run optimization
        self._study.optimize(_objective, n_trials=n_trials)

        return self.get_best_trial()

    def get_best_trial(self) -> dict[str, Any]:
        """Get the best trial from the Optuna study."""
        if self._study is None:
            raise RuntimeError("Study not created. Call create_study() first.")

        best_trial = self._study.best_trial

        return {
            "params": best_trial.params,
            "values": best_trial.values,
            "number": best_trial.number,
        }

    def save_study(self, config: TuningConfig) -> None:
        """Saves the study results in a csv file."""
        assert self._study
        self._study.trials_dataframe().to_csv(
            Path(config.tuning_params.output_dir, "trials_results.csv")
        )
        return super().save_study(config)
