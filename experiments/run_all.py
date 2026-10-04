"""Run the three experiments and regenerate every table and figure in results/.

Run:  python experiments/run_all.py [--quick]

The full run takes about 40 minutes on a 2-core CPU; --quick takes a couple of
minutes and writes to results/quick/ without touching the published results.
"""
import time

from common import parse_args

import gbm_frictionless
import gbm_transaction_costs
import heston_stochastic_vol


def main():
    settings = parse_args(__doc__.splitlines()[0])
    start = time.time()
    gbm_frictionless.main(settings)
    gbm_transaction_costs.main(settings)
    heston_stochastic_vol.main(settings)
    print(f"\nAll experiments finished in {(time.time() - start) / 60:.1f} min.")


if __name__ == "__main__":
    main()
