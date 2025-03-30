import os
import argparse
import json
from functools import partial
from time import time
from scipy.integrate import dblquad
import numpy as np
from numba import njit
from proportional_functions_opt import Psi


@njit
def laplace_Psi(
    n: int,
    c: list,
    alpha: list,
    poisson_rate: list,
    claims_rate: list,
    max_time: float,
    n_simulations,
    seed_offset: int,
    s: list,
    *args
) -> float:
    assert len(s) == len(args)
    exponent = np.exp(-sum([s[i] * args[i] for i in range(len(args))]))
    ruin_probability = Psi(
        u=args,
        n=n,
        c=c,
        alpha=alpha,
        poisson_rate=poisson_rate,
        claims_rate=claims_rate,
        max_time=max_time,
        n_simulations=n_simulations,
        seed_offset=seed_offset
    )
    return exponent * (1 - ruin_probability)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--n_simulations", type=int)
    parser.add_argument("--max_time", type=float)
    parser.add_argument("--seed_offset", type=int)
    args = parser.parse_args()

    config = json.load(fp=open(args.config))

    start_time = time()
    laplace_Psi_instance = partial(
        laplace_Psi,
        config["model_settings"]["n"],
        config["model_settings"]["c"],
        config["model_settings"]["alpha"],
        config["model_settings"]["poisson_rate"],
        config["model_settings"]["claims_rate"],
        args.max_time,
        args.n_simulations,
        args.seed_offset,
        config["s"]
    )
    int_res = dblquad(laplace_Psi_instance, 0, np.inf, 0, np.inf)

    result_dict = {"config": args.config, "n_simulations": args.n_simulations,
                   "max_time": args.max_time, "seed_offset": args.seed_offset,
                   "start_time": start_time, "end_time": time(),
                   "int_value": int_res[0], "abserr": int_res[1]}
    save_dir = "results/proportional_left"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    json.dump(obj=result_dict, fp=open(f"{save_dir}/result_{start_time}.json", "w"))

    print(f"int_value={result_dict['int_value']}; abserr={result_dict['abserr']}")
    print(f"time duration: {result_dict['end_time'] - result_dict['start_time']}")
