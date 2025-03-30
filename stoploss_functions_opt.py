from functools import partial
import numpy as np
from scipy.integrate import dblquad
from numba import njit


@njit
def stoploss_process_check_ruin(
                                time: float,
                                n: int,
                                u: list,
                                c: list,
                                b: list,
                                poisson_rate: list,
                                claims_rate: list,
                                seed: int = 42
                                ) -> bool:
    # base checks
    assert min(len(u), len(c), len(b) + 1, len(poisson_rate), len(claims_rate)) == n+1
    assert max(len(u), len(c), len(b) + 1, len(poisson_rate), len(claims_rate)) == n+1
    b = [-1] + b

    # set seed
    np.random.seed(seed=seed)

    claim_sizes_list = []
    inter_arrival_times_list = []
    # generate claims process until time
    for i in range(n+1):
        # batch_size = expectation of Poisson(\lambda * time)
        batch_size = int(poisson_rate[i] * time + 1)
        # generate arrival times
        inter_arrival_times = np.zeros(1)
        while inter_arrival_times[-1] <= time:
            batch_size *= 2
            inter_arrival_times = np.cumsum(np.random.exponential(1.0 / poisson_rate[i], size=batch_size))
        # generate claim sizes
        claim_sizes = np.random.exponential(1 / claims_rate[i], size=batch_size)

        # using bin search
        left = 0
        right = len(inter_arrival_times)
        while right - left > 1:
            mid = (right + left) // 2
            if inter_arrival_times[mid] <= time:
                left = mid
            else:
                right = mid
        if inter_arrival_times[left] > time:
            claim_idx_after_time = left
        else:
            claim_idx_after_time = right

        # cut inter_arrival_times and claim_sizes
        inter_arrival_times = inter_arrival_times[:claim_idx_after_time]
        claim_sizes = claim_sizes[:claim_idx_after_time]

        # save to list
        inter_arrival_times_list.append(inter_arrival_times)
        claim_sizes_list.append(claim_sizes)

    # concat claims
    claims_seq = [(inter_arrival_times_list[i_process][j], claim_sizes_list[i_process][j], i_process)
                  for i_process in range(n+1) for j in range(len(inter_arrival_times_list[i_process]))]

    claims_seq = sorted(claims_seq)

    # check ruin
    cumm_claim_sizes = np.zeros(n+1)
    for claim_time, claim_size, i_process in claims_seq:
        cumm_claim_sizes[i_process] += claim_size

        reins_claim_size = sum([max(cumm_claim_sizes[i] - b[i], 0) for i in range(1, n+1)])
        if u[0] + c[0] * claim_time - cumm_claim_sizes[0] - reins_claim_size < 0:
            return True
        if i_process > 0:
            if u[i_process] + c[i_process] * claim_time - cumm_claim_sizes[i_process] +\
                 max(cumm_claim_sizes[i_process] - b[i_process], 0) < 0:
                return True
    return False


@njit
def Psi(
    u: list,
    n: int,
    c: list,
    b: list,
    poisson_rate: list,
    claims_rate: list,
    max_time: float,
    n_simulations: int = 1000,
    seed_offset: int = 0
) -> float:
    is_ruin = np.full(n_simulations, False)
    for i_sim in range(n_simulations):
        is_ruin[i_sim] = stoploss_process_check_ruin(
            time=max_time,
            n=n,
            u=u,
            c=c,
            b=b,
            poisson_rate=poisson_rate,
            claims_rate=claims_rate,
            seed=i_sim + seed_offset + int(1e6 * sum(u))
        )
    return np.mean(is_ruin)


@njit
def laplace_Psi(
    n: int,
    c: list,
    b: list,
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
        b=b,
        poisson_rate=poisson_rate,
        claims_rate=claims_rate,
        max_time=max_time,
        n_simulations=n_simulations,
        seed_offset=seed_offset
    )
    return exponent * (1 - ruin_probability)


def calculate_laplace_transform(
    n: int,
    c: list,
    b: list,
    poisson_rate: list,
    claims_rate: list,
    max_time: float,
    n_simulations: int,
    seed_offset: int,
    s: list
):
    laplace_Psi_instance = partial(
        laplace_Psi,
        n,
        c,
        b,
        poisson_rate,
        claims_rate,
        max_time,
        n_simulations,
        seed_offset,
        s
    )
    return dblquad(laplace_Psi_instance, 0, np.inf, 0, np.inf)
