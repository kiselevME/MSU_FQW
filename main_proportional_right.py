import os
import argparse
import json
from functools import partial
from time import time
from scipy.integrate import quad
import numpy as np
from proportional_functions_opt import Psi
from common_functions import calculate_hat_pdf_exp


def int_Psi_function(
    n: int,
    c: list,
    alpha: list,
    poisson_rate: list,
    claims_rate: list,
    max_time: float,
    n_simulations,
    seed_offset: int,
    i: int,
    s: list,
    *args
) -> float:
    assert len(s) == len(args) + 1
    # zeroing i-th component
    args = list(args[:i]) + [0.0] + list(args[i:])
    exponent = np.exp(-sum([s[j] * args[j] for j in range(len(args)) if j != i]))

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


def calc_integral(
    n: int,
    c: list,
    alpha: list,
    poisson_rate: list,
    claims_rate: list,
    max_time: float,
    n_simulations,
    seed_offset: int,
    i: int,
    s: list
) -> float:
    int_Psi_function_instance = partial(
        int_Psi_function,
        n,
        c,
        alpha,
        poisson_rate,
        claims_rate,
        max_time,
        n_simulations,
        seed_offset,
        i,
        s
    )
    return quad(int_Psi_function_instance, 0, np.inf)[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--n_simulations", type=int)
    parser.add_argument("--max_time", type=float)
    parser.add_argument("--seed_offset", type=int)
    args = parser.parse_args()

    config = json.load(fp=open(args.config))

    start_time = time()

    # calculate numerator
    numerator = 0
    for i in range(config["model_settings"]["n"] + 1):
        int_value = calc_integral(
            n=config["model_settings"]["n"],
            c=config["model_settings"]["c"],
            alpha=config["model_settings"]["alpha"],
            poisson_rate=config["model_settings"]["poisson_rate"],
            claims_rate=config["model_settings"]["claims_rate"],
            max_time=args.max_time,
            n_simulations=args.n_simulations,
            seed_offset=args.seed_offset,
            i=i,
            s=config["s"]
        )
        numerator += config["model_settings"]["c"][i] * int_value
    # calculate denominator
    denominator = 0
    for i in range(config["model_settings"]["n"] + 1):
        denominator += config["model_settings"]["c"][i] * config["s"][i]
        if i == 0:
            int_value = calculate_hat_pdf_exp(s=config["s"][0], lambd=config["model_settings"]["poisson_rate"][0])
        else:
            int_value = calculate_hat_pdf_exp(
                s=config["s"][0] * (1 - config["model_settings"]["alpha"][i-1]) +
                config["s"][i] * config["model_settings"]["alpha"][i-1],
                lambd=config["model_settings"]["poisson_rate"][i]
            )
        denominator += config["model_settings"]["poisson_rate"][i] * (int_value - 1)

    result_dict = {"config": args.config, "n_simulations": args.n_simulations,
                   "max_time": args.max_time, "seed_offset": args.seed_offset,
                   "start_time": start_time, "end_time": time(),
                   "value": numerator / denominator}
    save_dir = "results/proportional_right"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    json.dump(obj=result_dict, fp=open(f"{save_dir}/result_{start_time}.json", "w"))

    print(f"value={result_dict['value']}")
    print(f"time duration: {result_dict['end_time'] - result_dict['start_time']}")
