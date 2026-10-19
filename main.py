"""
Single entry point for every method.

    python main.py --method random --env kuhn_poker   --budget 100000 --seed 0
    python main.py --method psro   --env leduc_poker  --budget 1000000 --seed 0

Normally invoked through run.sh.

main.py builds the pieces every method shares -- environment factory, budget,
logger, evaluator, RNG -- and hands them to the method named by --method. At the
same --budget every method gets the same number of environment steps, the same
evaluation watermarks and the same metrics, so their eval.jsonl files can be
plotted on one axis without any post-hoc alignment.

Output: results/<exp-name>/<env>/<method>/seed<seed>/{config.json, eval.jsonl, train.jsonl}
"""

import os

import numpy as np

from config import get_config, resolve, save_config
from seprso.budget import BudgetTracker
from seprso.envs import make_env
from seprso.evaluation import Evaluator
from seprso.logging import RunLogger
from seprso.registry import METHODS
from seprso.utils import set_seed


def main(argv=None):
    args = resolve(get_config().parse_args(argv))
    set_seed(args.seed)

    def env_fn():
        return make_env(args.env_preset, max_episode_steps=args.max_episode_steps,
                        obs_type=args.obs_type)

    probe = env_fn()
    spec = probe.spec
    probe.close()

    os.makedirs(args.run_dir, exist_ok=True)
    save_config(args, os.path.join(args.run_dir, 'config.json'))
    # Everything that defines an experimental arm goes into meta, so it is on
    # every eval record and the plotter can tell arms apart.
    logger = RunLogger(args.run_dir, meta={'method': args.method, 'env': args.env_name,
                                           'seed': args.seed, 'budget': args.budget})

    budget = BudgetTracker(args.budget)
    evaluator = Evaluator(env_fn, budget, logger, metrics=args.eval_metrics,
                          interval=args.eval_interval, episodes=args.eval_episodes,
                          seed=args.seed)

    method_cls = METHODS.get(args.method)
    method = method_cls(args=args, env_fn=env_fn, spec=spec, budget=budget, logger=logger,
                        evaluator=evaluator, rng=np.random.default_rng([args.seed, 0]))
    print(f'[seprso] {args.method} on {args.env_name} ({spec.backend}, '
          f'{spec.num_agents} agents, {"turn-based" if spec.turn_based else "simultaneous"}) '
          f'budget={args.budget} seed={args.seed} -> {args.run_dir}', flush=True)

    try:
        method.train()
        evaluator.evaluate(method, final=True)
    finally:
        evaluator.close()
        logger.close()


if __name__ == '__main__':
    main()
