import os
import argparse
import json
from time import time
from stoploss_functions_opt import calculate_laplace_transform


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--n_simulations", type=int)
    parser.add_argument("--max_time", type=float)
    parser.add_argument("--seed_offset", type=int)
    args = parser.parse_args()

    config = json.load(fp=open(args.config))

    start_time = time()

    int_res = calculate_laplace_transform(
        n=config["model_settings"]["n"],
        c=config["model_settings"]["c"],
        b=config["model_settings"]["b"],
        poisson_rate=config["model_settings"]["poisson_rate"],
        claims_rate=config["model_settings"]["claims_rate"],
        max_time=args.max_time,
        n_simulations=args.n_simulations,
        seed_offset=args.seed_offset,
        s=config["s"]
    )

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
