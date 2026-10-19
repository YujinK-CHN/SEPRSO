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
    psro / ppo / gems
                method hyper-parameters -- empty until each method is written
"""

import argparse
import json
import os

from seprso.registry import METHODS, METRICS
import seprso.methods  # noqa: F401  (registers methods)
import seprso.evaluation  # noqa: F401  (registers metrics)


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
    parser.add_argument_group('psro')


def _add_ppo_config(parser):
    parser.add_argument_group('ppo')


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
