"""
Single configuration file for every method in SEPRSO.

Only a few things are meant to change from the command line, because only they
change between runs of an experiment suite:

    --method    which method       (see seprso/methods; `--help` lists them)
    --env       which environment  (a key of ENV_PRESETS, or backend:id)
    --budget    total training budget, in ENV STEPS (see seprso/budget.py)
    --seed      random seed

    python main.py --method random --env kuhn_poker --budget 100000 --seed 0

Everything else -- evaluation protocol, every method's hyper-parameters -- is
set by the defaults below. Edit them here. They are still CLI flags so a one-off
sweep can override one without editing the file, but run.sh does not pass them,
so this file stays the single source of truth.

Groups:
    run         the knobs above, plus where output goes
    env         environment options shared by all backends
    eval        evaluation protocol (identical for every method)
    psro        PSRO loop (names follow OpenSpiel's psro_v2)
    ppo         PPO learner, used by PSRO's best-response oracle
    gems        empty until GEMS is written
"""

import argparse
import json
import os

from seprso.meta_solvers import META_SOLVERS
from seprso.registry import METHODS, METRICS
import seprso.methods  # noqa: F401  (registers methods)
import seprso.evaluation  # noqa: F401  (registers metrics)
import seprso.exploitability  # noqa: F401  (registers nash_conv)


# --------------------------------------------------------------------------- #
# Environments.  Keyed by --env.
#
#   backend   'native' | 'openspiel' | 'pettingzoo'
#   id        matrix-game name | OpenSpiel game name | PettingZoo module path
#   api       PettingZoo only: 'aec' (turn-based) or 'parallel' (simultaneous)
#   kwargs    OpenSpiel game parameters / PettingZoo env kwargs
#
# Any environment not listed can still be used ad hoc as --env backend:id, e.g.
#   --env openspiel:liars_dice
#   --env pettingzoo:pettingzoo.classic.go_v5
#   --env pettingzoo:mpe2.simple_tag_v3@parallel
# Add it here once it is used for real, so its options are recorded.
# --------------------------------------------------------------------------- #
ENV_PRESETS = {
    # native -- no dependencies; for debugging methods
    'rps':              {'backend': 'native', 'id': 'rps'},
    'matching_pennies': {'backend': 'native', 'id': 'matching_pennies'},
    'prisoners_dilemma': {'backend': 'native', 'id': 'prisoners_dilemma'},

    # OpenSpiel -- imperfect-information benchmarks of the PSRO literature
    'kuhn_poker':       {'backend': 'openspiel', 'id': 'kuhn_poker'},
    'leduc_poker':      {'backend': 'openspiel', 'id': 'leduc_poker'},
    'goofspiel':        {'backend': 'openspiel', 'id': 'goofspiel',
                         'kwargs': {'num_cards': 5, 'points_order': 'descending'}},
    'liars_dice':       {'backend': 'openspiel', 'id': 'liars_dice'},
    'os_tic_tac_toe':   {'backend': 'openspiel', 'id': 'tic_tac_toe'},

    # PettingZoo
    'connect_four':     {'backend': 'pettingzoo', 'id': 'pettingzoo.classic.connect_four_v3'},
    'tic_tac_toe':      {'backend': 'pettingzoo', 'id': 'pettingzoo.classic.tictactoe_v3'},
    'pz_rps':           {'backend': 'pettingzoo', 'id': 'pettingzoo.classic.rps_v2'},
}


def parse_env(value):
    """--env value -> preset dict. Accepts a preset key or 'backend:id[@api]'."""
    if value in ENV_PRESETS:
        return {'name': value, **ENV_PRESETS[value]}
    if ':' not in value:
        raise ValueError(f'Unknown env {value!r}. Use one of {sorted(ENV_PRESETS)} '
                         'or the form backend:id (e.g. openspiel:liars_dice).')
    backend, ident = value.split(':', 1)
    preset = {'backend': backend, 'id': ident}
    if '@' in ident:
        preset['id'], preset['api'] = ident.split('@', 1)
    preset['name'] = preset['id'].rsplit('.', 1)[-1]
    return preset


def _add_run_config(parser):
    group = parser.add_argument_group('run')
    group.add_argument('--method', type=str, default='random', choices=METHODS.names())
    group.add_argument('--env', type=str, default='kuhn_poker',
                       help=f'One of {sorted(ENV_PRESETS)}, or backend:id.')
    group.add_argument('--budget', type=int, default=100_000,
                       help='Total training budget in ENV STEPS (calls to env.step). '
                            'The common currency that makes methods comparable.')
    group.add_argument('--seed', type=int, default=0)
    group.add_argument('--exp-name', type=str, default='check',
                       help='Groups runs under results/<exp-name>/.')
    group.add_argument('--results-dir', type=str, default='results')
    group.add_argument('--device', type=str, default='auto',
                       help="'auto' picks cuda when available, else cpu.")


def _add_env_config(parser):
    group = parser.add_argument_group('env')
    group.add_argument('--max-episode-steps', type=int, default=None,
                       help='Truncate episodes after this many env steps (None = game length).')
    group.add_argument('--obs-type', type=str, default='info_state',
                       choices=('info_state', 'observation'),
                       help='OpenSpiel only: perfect-recall information state or Markov observation.')


def _add_eval_config(parser):
    group = parser.add_argument_group('eval')
    group.add_argument('--eval-interval', type=int, default=None,
                       help='Evaluate every this many env steps. Default: budget / 20.')
    group.add_argument('--eval-episodes', type=int, default=200)
    group.add_argument('--eval-metrics', type=str, nargs='+',
                       default=['vs_random', 'self_play'], choices=METRICS.names())


def _add_psro_config(parser):
    # Names follow OpenSpiel's psro_v2 where the concept is the same.
    group = parser.add_argument_group('psro')
    group.add_argument('--psro-meta-solver', type=str, default='auto',
                       choices=META_SOLVERS.names(),
                       help="'auto' = nash for two-player zero-sum tables, else prd.")
    group.add_argument('--psro-training-selector', type=str, default='probabilistic',
                       choices=('probabilistic', 'uniform'),
                       help='What best responses train against: the meta-strategy, '
                            'or every population member equally.')
    group.add_argument('--psro-sims-per-entry', type=int, default=100,
                       help='Episodes simulated per payoff-table entry (charged to budget).')
    group.add_argument('--psro-br-steps', type=int, default=20_000,
                       help='Env steps of PPO training per best response.')
    group.add_argument('--psro-br-init', type=str, default='scratch',
                       choices=('scratch', 'latest'),
                       help="Start each best response from scratch, or from the seat's "
                            'latest response.')
    group.add_argument('--psro-prd-iterations', type=int, default=100_000)
    group.add_argument('--psro-prd-dt', type=float, default=1e-3)
    group.add_argument('--psro-prd-gamma', type=float, default=1e-6,
                       help='PRD exploration floor: minimum mass on every member.')


def _add_ppo_config(parser):
    # Shared by the PSRO best-response oracle and the (future) ppo method.
    # Defaults are CleanRL's, except a smaller rollout for short board/card games.
    group = parser.add_argument_group('ppo')
    group.add_argument('--ppo-lr', type=float, default=2.5e-4)
    group.add_argument('--ppo-rollout-steps', type=int, default=1024,
                       help="Learner transitions (the learning seat's decisions) per update.")
    group.add_argument('--ppo-epochs', type=int, default=4)
    group.add_argument('--ppo-minibatches', type=int, default=4)
    group.add_argument('--ppo-gamma', type=float, default=0.99)
    group.add_argument('--ppo-gae-lambda', type=float, default=0.95)
    group.add_argument('--ppo-clip', type=float, default=0.2)
    group.add_argument('--ppo-ent-coef', type=float, default=0.01)
    group.add_argument('--ppo-vf-coef', type=float, default=0.5)
    group.add_argument('--ppo-max-grad-norm', type=float, default=0.5)
    group.add_argument('--ppo-hidden', type=int, nargs='+', default=[128, 128])


def _add_gems_config(parser):
    parser.add_argument_group('gems')


def get_config():
    parser = argparse.ArgumentParser(
        description='SEPRSO', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    _add_run_config(parser)
    _add_env_config(parser)
    _add_eval_config(parser)
    _add_psro_config(parser)
    _add_ppo_config(parser)
    _add_gems_config(parser)
    return parser


def resolve(args):
    """Fill in derived fields. Everything downstream reads only the result."""
    args.env_preset = parse_env(args.env)
    args.env_name = args.env_preset['name']
    if args.eval_interval is None:
        args.eval_interval = max(1, args.budget // 20)
    if args.device == 'auto':
        try:
            import torch
            args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        except ImportError:
            args.device = 'cpu'
    args.run_dir = os.path.join(args.results_dir, args.exp_name, args.env_name,
                                args.method, f'seed{args.seed}')
    return args


def save_config(args, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(vars(args), handle, indent=2, sort_keys=True, default=str)
