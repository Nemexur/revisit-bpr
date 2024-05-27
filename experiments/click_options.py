from typing import Any, Callable, Optional
from dataclasses import dataclass, field
from pathlib import Path
import re

import click


class DictParamType(click.types.ParamType):
    """A Click type to represent dictionary as parameters for command."""

    name = "DICT"

    def __init__(self, value_type: Callable = str, **kwargs) -> None:
        super().__init__(**kwargs)
        self._value_type = value_type

    def convert(
        self,
        value: str,
        param: Optional[click.core.Parameter],
        ctx: Optional[click.core.Context],
    ) -> Optional[dict[str, Any]]:
        """
        Convert value to an appropriate representation.

        Parameters
        ----------
        value: str
            Value assigned to a parameter.
        param: Optional[click.core.Parameter]
            Parameter with assigned value.
        ctx: Optional[click.core.Context]
            Context for CLI.

        Returns
        -------
        Dict[str, Any]
            Key-Value dictionary of parameters.
        """
        extra_vars = super().convert(value=value, param=param, ctx=ctx)
        regex = r"([a-z0-9\_\-\.\+\\\/]+)=([a-z0-9:\_\-\.\+\\\/]+)"
        return (
            {
                param: self._value_type(value)
                for param, value in re.findall(regex, extra_vars, flags=re.I)
            }
            if extra_vars is not None
            else None
        )


@dataclass
class EarlyStopping:
    metric: str | None = None
    patience: int = 200
    direction: str = "max"


@dataclass
class SearchHP:
    run: bool = False
    metric: str | None = None
    storage: str | None = None
    trials: int = 10
    seed: int = 13
    train_best: bool = True
    prune: bool = False


@dataclass
class State:
    exp_name: str = "exp"
    exp_dir: Path | None = None
    seed: int = 13
    debug: bool = False
    use_wandb: bool = False
    use_clearml: bool = False
    extra_vars: dict[str, Any] | None = None
    search_hp: SearchHP = field(default_factory=SearchHP)
    early_stopping: EarlyStopping = field(default_factory=EarlyStopping)


pass_state = click.make_pass_decorator(State, ensure=True)


def name_option(default: str = "exp") -> Callable:
    """
    Add name option to CLI command.

    Parameters
    ----------
    default: str | None (default = None)
        Experiment name.

    Returns
    -------
    Callable
        Click command/group with new option.
    """

    def wrapper(f: Callable) -> Callable:
        def callback(ctx: click.Context, _: click.core.Parameter, value: str) -> Any:
            state: State = ctx.ensure_object(State)
            state.exp_name = value
            return value

        return click.option(
            "-n",
            "--name",
            type=click.STRING,
            help="Experiment name.",
            callback=callback,
            expose_value=False,
            required=False,
            default=default,
            show_default=True,
        )(f)

    return wrapper


def dir_option(f: Callable) -> Callable:
    """
    Add dir option to CLI command.

    Parameters
    ----------
    f: Callable
        Click command/group.

    Returns
    -------
    Callable
        Click command/group with new option.
    """

    def callback(ctx: click.Context, _: click.core.Parameter, value: Path) -> Any:
        state: State = ctx.ensure_object(State)
        state.exp_dir = value
        return value

    return click.option(
        "-d",
        "--dir",
        type=click.Path(exists=False, path_type=Path),
        help="Experiment directory.",
        callback=callback,
        expose_value=False,
        required=False,
        default=None,
        show_default=True,
    )(f)


def debug_option(f: Callable) -> Callable:
    """
    Add debug option to CLI command.

    Parameters
    ----------
    f: Callable
        Click command/group.

    Returns
    -------
    Callable
        Click command/group with new option.
    """

    def callback(ctx: click.Context, _: click.core.Parameter, value: bool) -> Any:
        state: State = ctx.ensure_object(State)
        state.debug = value
        return value

    return click.option(
        "--debug",
        is_flag=True,
        help="Run experiment in debug mode.",
        callback=callback,
        expose_value=False,
        required=False,
    )(f)


def seed_option(f: Callable) -> Callable:
    """
    Add seed option to CLI command.

    Parameters
    ----------
    f: Callable
        Click command/group.

    Returns
    -------
    Callable
        Click command/group with new option.
    """

    def callback(ctx: click.Context, _: click.core.Parameter, value: int) -> Any:
        state: State = ctx.ensure_object(State)
        state.seed = value
        return value

    return click.option(
        "--seed",
        type=click.INT,
        callback=callback,
        expose_value=False,
        required=False,
        default=13,
        show_default=True,
    )(f)


def wandb_option(f: Callable) -> Callable:
    """
    Add wandb option to CLI command.

    Parameters
    ----------
    f: Callable
        Click command/group.

    Returns
    -------
    Callable
        Click command/group with new option.
    """

    def callback(ctx: click.Context, _: click.core.Parameter, value: bool) -> Any:
        state: State = ctx.ensure_object(State)
        state.use_wandb = value
        return value

    return click.option(
        "--wandb",
        is_flag=True,
        help="Whether to enable wandb for the experiment or not.",
        callback=callback,
        expose_value=False,
        required=False,
    )(f)


def clearml_option(f: Callable) -> Callable:
    """
    Add clearml option to CLI command.

    Parameters
    ----------
    f: Callable
        Click command/group.

    Returns
    -------
    Callable
        Click command/group with new option.
    """

    def callback(ctx: click.Context, _: click.core.Parameter, value: bool) -> Any:
        state: State = ctx.ensure_object(State)
        state.use_clearml = value
        return value

    return click.option(
        "--clearml",
        is_flag=True,
        help="Whether to enable clearml for the experiment or not.",
        callback=callback,
        expose_value=False,
        required=False,
    )(f)


def extra_vars_option(f: Callable) -> Callable:
    """
    Add extra-vars option to CLI command.

    Parameters
    ----------
    f: Callable
        Click command/group.

    Returns
    -------
    Callable
        Click command/group with new option.
    """

    def callback(ctx: click.Context, _: click.core.Parameter, value: dict[str, Any]) -> Any:
        state: State = ctx.ensure_object(State)
        state.extra_vars = value
        return value

    return click.option(
        "--extra-vars",
        type=DictParamType(),
        help=(
            "Extra variables to inject to yaml config. "
            "Format: {key_name1}={new_value1},{key_name2}={new_value2},..."
        ),
        callback=callback,
        expose_value=False,
        required=False,
        default=None,
        show_default=True,
    )(f)


def search_hp_option(f: Callable) -> Callable:
    """
    Add search-hp option to CLI command.

    Parameters
    ----------
    f: Callable
        Click command/group.

    Returns
    -------
    Callable
        Click command/group with new option.
    """

    def callback(ctx: click.Context, _: click.core.Parameter, value: bool) -> Any:
        state: State = ctx.ensure_object(State)
        state.search_hp.run = value
        return value

    return click.option(
        "--search-hp",
        is_flag=True,
        help="Whether to search for hyperparameters or not.",
        callback=callback,
        expose_value=False,
        required=False,
    )(f)


def search_hp_metric_option(name: Optional[str] = None) -> Callable:
    """
    Add search-hp-metric option to CLI command.

    Parameters
    ----------
    name: Optional[str] (default = None)
        Metric to monitor for HP search.

    Returns
    -------
    Callable
        Click command/group with new option.
    """

    def wrapper(f: Callable) -> Callable:
        def callback(ctx: click.Context, _: click.core.Parameter, value: str) -> Any:
            state: State = ctx.ensure_object(State)
            state.search_hp.metric = value
            return value

        return click.option(
            "--search-hp-metric",
            type=click.STRING,
            help="Main metric for hyperparameters optimization.",
            callback=callback,
            expose_value=False,
            required=False,
            default=name,
            show_default=True,
        )(f)

    return wrapper


def search_hp_storage_option(f: Callable) -> Callable:
    """
    Add search-hp-storage option to CLI command.

    Parameters
    ----------
    f: Callable
        Click command/group.

    Returns
    -------
    Callable
        Click command/group with new option.
    """

    def callback(ctx: click.Context, _: click.core.Parameter, value: str) -> Any:
        state: State = ctx.ensure_object(State)
        state.search_hp.storage = value
        return value

    return click.option(
        "--search-hp-storage",
        type=click.STRING,
        help=(
            "Storage url for optuna RDB. "
            "If None optuna uses either in-memory or sqlite depending on dir option."
        ),
        callback=callback,
        expose_value=False,
        required=False,
        default=None,
        show_default=True,
        envvar="SEARCH_HP_STORAGE",
    )(f)


def search_hp_trials_option(f: Callable) -> Callable:
    """
    Add search-hp-trials option to CLI command.

    Parameters
    ----------
    f: Callable
        Click command/group.

    Returns
    -------
    Callable
        Click command/group with new option.
    """

    def callback(ctx: click.Context, _: click.core.Parameter, value: int) -> Any:
        state: State = ctx.ensure_object(State)
        state.search_hp.trials = value
        return value

    return click.option(
        "--search-hp-trials",
        type=click.INT,
        help="Number of trials for hyperparameters search.",
        callback=callback,
        expose_value=False,
        required=False,
        default=10,
        show_default=True,
    )(f)


def search_hp_seed_option(f: Callable) -> Callable:
    """
    Add search-hp-seed option to CLI command.

    Parameters
    ----------
    f: Callable
        Click command/group.

    Returns
    -------
    Callable
        Click command/group with new option.
    """

    def callback(ctx: click.Context, _: click.core.Parameter, value: int) -> Any:
        state: State = ctx.ensure_object(State)
        state.search_hp.seed = value
        return value

    return click.option(
        "--search-hp-seed",
        type=click.INT,
        callback=callback,
        expose_value=False,
        required=False,
        default=13,
        show_default=True,
    )(f)


def search_hp_train_best_option(f: Callable) -> Callable:
    """
    Add search-hp-train-best option to CLI command.

    Parameters
    ----------
    f: Callable
        Click command/group.

    Returns
    -------
    Callable
        Click command/group with new option.
    """

    def callback(ctx: click.Context, _: click.core.Parameter, value: bool) -> Any:
        state: State = ctx.ensure_object(State)
        state.search_hp.train_best = value
        return value

    return click.option(
        "--search-hp-train-best",
        is_flag=True,
        help="Whether to train best model or not.",
        callback=callback,
        expose_value=False,
        required=False,
    )(f)


def search_hp_prune_option(f: Callable) -> Callable:
    """
    Add search-hp-prune option to CLI command.

    Parameters
    ----------
    f: Callable
        Click command/group.

    Returns
    -------
    Callable
        Click command/group with new option.
    """

    def callback(ctx: click.Context, _: click.core.Parameter, value: bool) -> Any:
        state: State = ctx.ensure_object(State)
        state.search_hp.prune = value
        return value

    return click.option(
        "--search-hp-prune",
        is_flag=True,
        help="Whether to enable prunning or not.",
        callback=callback,
        expose_value=False,
        required=False,
    )(f)


def early_stopping_metric_option(name: Optional[str] = None) -> Callable:
    """
    Add early-stopping-metric option to CLI command.

    Parameters
    ----------
    name: Optional[str] (default = None)
        Default metric for early stopping.

    Returns
    -------
    Callable
        Click command/group with new option.
    """

    def wrapper(f: Callable) -> Callable:
        def callback(ctx: click.Context, _: click.core.Parameter, value: str) -> Any:
            state: State = ctx.ensure_object(State)
            state.early_stopping.metric = value
            return value

        return click.option(
            "--early-stopping-metric",
            type=click.STRING,
            help="Metric for early stopping.",
            callback=callback,
            expose_value=False,
            required=False,
            default=name,
            show_default=True,
        )(f)

    return wrapper


def early_stopping_patience_option(f: Callable) -> Callable:
    """
    Add early-stopping-patience option to CLI command.

    Parameters
    ----------
    f: Callable
        Click command/group.

    Returns
    -------
    Callable
        Click command/group with new option.
    """

    def callback(ctx: click.Context, _: click.core.Parameter, value: int) -> Any:
        state: State = ctx.ensure_object(State)
        state.early_stopping.patience = value
        return value

    return click.option(
        "--early-stopping-patience",
        type=click.INT,
        help="Patience iterations for early stopping.",
        callback=callback,
        expose_value=False,
        required=False,
        default=200,
        show_default=True,
    )(f)


def early_stopping_direction_option(f: Callable) -> Callable:
    """
    Add early-stopping-direction option to CLI command.

    Parameters
    ----------
    f: Callable
        Click command/group.

    Returns
    -------
    Callable
        Click command/group with new option.
    """

    def callback(ctx: click.Context, _: click.core.Parameter, value: str) -> Any:
        state: State = ctx.ensure_object(State)
        state.early_stopping.direction = value
        return value

    return click.option(
        "--early-stopping-direction",
        type=click.Choice(["min", "max"], case_sensitive=False),
        help="Metric direction for early stopping.",
        callback=callback,
        expose_value=False,
        required=False,
        default="max",
        show_default=True,
    )(f)


def search_hp_options(metric: Optional[str] = None) -> Callable:
    """
    Add options:
    - search-hp
    - search-hp-metric
    - search-hp-storage
    - search-hp-trials
    - search-hp-seed
    - search-hp-train-best
    - search-hp-prune

    Parameters
    ----------
    f: Callable
        Click command/group.

    Returns
    -------
    Callable
        Passed click command/group with additional options and arguments.
    """

    def wrapper(f: Callable) -> Callable:
        f = search_hp_option(f)
        f = search_hp_metric_option(metric)(f)
        f = search_hp_storage_option(f)
        f = search_hp_trials_option(f)
        f = search_hp_seed_option(f)
        f = search_hp_train_best_option(f)
        f = search_hp_prune_option(f)
        return f

    return wrapper


def early_stopping_options(metric: Optional[str] = None) -> Callable:
    """
    Add options:
    - early-stopping-metric
    - early-stopping-patience
    - early-stopping-direction

    Parameters
    ----------
    f: Callable
        Click command/group.

    Returns
    -------
    Callable
        Passed click command/group with additional options and arguments.
    """

    def wrapper(f: Callable) -> Callable:
        f = early_stopping_metric_option(metric)(f)
        f = early_stopping_patience_option(f)
        f = early_stopping_direction_option(f)
        return f

    return wrapper
