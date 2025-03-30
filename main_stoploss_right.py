import os
import argparse
import json
from functools import partial
from copy import copy
from time import time
from scipy.integrate import quad
import numpy as np
from stoploss_functions_opt import Psi, calculate_laplace_transform
from common_functions import pdf_exp, calculate_hat_pdf_exp


def int_Psi_function(
    n: int,
    c: list,
    b: list,
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
        b=b,
        poisson_rate=poisson_rate,
        claims_rate=claims_rate,
        max_time=max_time,
        n_simulations=n_simulations,
        seed_offset=seed_offset
    )
    return exponent * (1 - ruin_probability)


def calc_integral_1(
    n: int,
    c: list,
    b: list,
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
        b,
        poisson_rate,
        claims_rate,
        max_time,
        n_simulations,
        seed_offset,
        i,
        s
    )
    return quad(int_Psi_function_instance, 0, np.inf)[0]


def int_laplace_Psi_function(
    n: int,
    c: list,
    b: list,
    poisson_rate: list,
    claims_rate: list,
    max_time: float,
    n_simulations,
    seed_offset: int,
    i: int,
    s: list,
    x: float
) -> float:
    mod_b = copy(b)
    mod_b[i-1] -= x

    laplace_bar_Psi = calculate_laplace_transform(
        n=n,
        c=c,
        b=mod_b,
        poisson_rate=poisson_rate,
        claims_rate=claims_rate,
        max_time=max_time,
        n_simulations=n_simulations,
        seed_offset=seed_offset,
        s=s
    )
    exponent = np.exp(-s[i] * x)
    pdf = pdf_exp(lambd=poisson_rate[i], x=x)

    return laplace_bar_Psi[0] * exponent * pdf


def calc_integral_2(
    n: int,
    c: list,
    b: list,
    poisson_rate: list,
    claims_rate: list,
    max_time: float,
    n_simulations,
    seed_offset: int,
    i: int,
    s: list
) -> float:
    int_laplace_Psi_function_instance = partial(
        int_laplace_Psi_function,
        n,
        c,
        b,
        poisson_rate,
        claims_rate,
        max_time,
        n_simulations,
        seed_offset,
        i,
        s
    )
    return quad(int_laplace_Psi_function_instance, 0, b[i-1])[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--n_simulations", type=int)
    parser.add_argument("--max_time", type=float)
    parser.add_argument("--seed_offset", type=int)
    args = parser.parse_args()

    config = json.load(fp=open(args.config))
    n = config["model_settings"]["n"]
    c = config["model_settings"]["c"]
    b = config["model_settings"]["b"]
    poisson_rate = config["model_settings"]["poisson_rate"]
    claims_rate = config["model_settings"]["claims_rate"]
    s = config["s"]
    n_simulations = args.n_simulations
    max_time = args.max_time
    seed_offset = args.seed_offset

    start_time = time()

    # calculate numerator
    numerator = 0
    for i in range(n + 1):
        int_value = calc_integral_1(
            n=n,
            c=c,
            b=b,
            poisson_rate=poisson_rate,
            claims_rate=claims_rate,
            max_time=max_time,
            n_simulations=n_simulations,
            seed_offset=seed_offset,
            i=i,
            s=s
        )
        numerator += c[i] * int_value
        if i > 0:
            int_value = calc_integral_2(
                n=n,
                c=c,
                b=b,
                poisson_rate=poisson_rate,
                claims_rate=claims_rate,
                max_time=max_time,
                n_simulations=n_simulations,
                seed_offset=seed_offset,
                i=i,
                s=s
            )

            exponent = np.exp((s[0] - s[i]) * b[i-1])
            mod_b = copy(b)
            mod_b[i-1] = 0
            laplace_bar_Psi = calculate_laplace_transform(
                n=n,
                c=c,
                b=mod_b,
                poisson_rate=poisson_rate,
                claims_rate=claims_rate,
                max_time=max_time,
                n_simulations=n_simulations,
                seed_offset=seed_offset,
                s=s
            )
            hat_pdf = calculate_hat_pdf_exp(s=s[0], lambd=poisson_rate[i], lower_limit=b[i-1])

            numerator -= poisson_rate[i] * (int_value - exponent * laplace_bar_Psi * hat_pdf)

    # calculate denominator
    denominator = 0
    for i in range(n + 1):
        denominator += c[i] * s[i]
        if i == 0:
            int_value = calculate_hat_pdf_exp(s=s[0], lambd=poisson_rate[0])
            denominator += poisson_rate[0] * (int_value - 1)
        else:
            denominator -= poisson_rate[i]

    result_dict = {"config": args.config, "n_simulations": n_simulations,
                   "max_time": max_time, "seed_offset": seed_offset,
                   "start_time": start_time, "end_time": time(),
                   "value": numerator / denominator}
    save_dir = "results/proportional_right"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    json.dump(obj=result_dict, fp=open(f"{save_dir}/result_{start_time}.json", "w"))

    print(f"value={result_dict['value']}")
    print(f"time duration: {result_dict['end_time'] - result_dict['start_time']}")
