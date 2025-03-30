import numpy as np
import pandas as pd
import plotly.express as px


class X_t_proportional:
    """
    Process X(t) = (X_0(t), X_1(t), ..., X_n(t)).
    """
    def __init__(self, n: int, u: list, c: list, alpha: list, poisson_rate: list, claims_rate: list, seed: int = 42):
        # base checks
        assert min(len(u), len(c), len(alpha) + 1, len(poisson_rate), len(claims_rate)) == n+1
        assert min(len(u), len(c), len(alpha) + 1, len(poisson_rate), len(claims_rate)) == n+1

        self.n = n
        self.u = u
        self.c = c
        self.alpha = [-1] + alpha
        self.poisson_rate = poisson_rate
        self.claims_rate = claims_rate
        self.seed = seed
        self.inter_arrival_times = [[] for _ in range(n+1)]
        self.claim_sizes = [[] for _ in range(n+1)]

    def get_value_at_time(self, time: float) -> list:
        value = [0 for _ in range(self.n+1)]
        cumm_claim_sizes = []
        # generate claims process until time
        for i in range(0, self.n+1):
            while (self.inter_arrival_times[i] == []) or\
                  (self.inter_arrival_times[i][-1] <= time):
                self._generate_next_claim(i_process=i)
            claim_idx_after_time = self._find_first_claim_idx_after_time_moment(i_process=i, time=time)
            cumm_claim_size = sum(self.claim_sizes[i][:claim_idx_after_time])
            cumm_claim_sizes.append(cumm_claim_size)
        # 1,2,...,n processes
        for i in range(1, self.n+1):
            value[i] = self.u[i] + self.c[i] * time - self.alpha[i] * cumm_claim_sizes[i]
        # 0 process
        value[0] = self.u[0] + self.c[0] * time - cumm_claim_sizes[0] - sum([(1 - self.alpha[i]) * cumm_claim_sizes[i]
                                                                             for i in range(1, self.n+1)])
        return value

    def _generate_next_claim(self, i_process: int):
        self._generate_next_arrival_time(i_process)
        self._generate_next_claim_size(i_process)

    def _generate_next_arrival_time(self, i_process: int):
        if self.inter_arrival_times[i_process]:
            # set seed and generate
            np.random.seed(seed=int(self.seed * (1 + 1e6 * self.inter_arrival_times[i_process][-1]) % 2**32))
            inter_arrival_time = np.random.exponential(1.0 / self.poisson_rate[i_process])
            last_arrival_time = self.inter_arrival_times[i_process][-1] + inter_arrival_time
        else:
            # set seed and generate
            np.random.seed(seed=int(self.seed * (1 + 1e7 * i_process) % 2**32))
            inter_arrival_time = np.random.exponential(1.0 / self.poisson_rate[i_process])
            last_arrival_time = inter_arrival_time
        self.inter_arrival_times[i_process].append(last_arrival_time)

    def _generate_next_claim_size(self, i_process: int):
        claim_size = np.random.exponential(1 / self.claims_rate[i_process])
        self.claim_sizes[i_process].append(claim_size)

    def _find_first_claim_idx_after_time_moment(self, i_process: int, time: float) -> int:
        # using bin search
        left = 0
        right = len(self.inter_arrival_times[i_process])
        while right - left > 1:
            mid = (right + left) // 2
            if self.inter_arrival_times[i_process][mid] <= time:
                left = mid
            else:
                right = mid
        if self.inter_arrival_times[i_process][left] > time:
            return left
        else:
            return right

    def plot_process(self, max_time: float, n_time_points: int = 1000):
        t = np.linspace(0, max_time, n_time_points)
        y = np.empty((self.n+1, n_time_points))
        for time_idx, time in enumerate(t):
            value = self.get_value_at_time(time=time)
            for i, process_value in enumerate(value):
                y[i, time_idx] = process_value
        df = pd.DataFrame(np.vstack([t, y]).T, columns=['time'] + [f'{i} company' for i in range(0, self.n+1)])
        return px.line(df, x='time', y=[f'{i} company' for i in range(0, self.n+1)])


class X_t_stoploss:
    """
    Process X(t) = (X_0(t), X_1(t), ..., X_n(t)).
    """
    def __init__(self, n: int, u: list, c: list, b: list, poisson_rate: list, claims_rate: list, seed: int = 42):
        # base checks
        assert min(len(u), len(c), len(b) + 1, len(poisson_rate), len(claims_rate)) == n+1
        assert min(len(u), len(c), len(b) + 1, len(poisson_rate), len(claims_rate)) == n+1

        self.n = n
        self.u = u
        self.c = c
        self.b = [-1] + b
        self.poisson_rate = poisson_rate
        self.claims_rate = claims_rate
        self.seed = seed
        self.inter_arrival_times = [[] for _ in range(n+1)]
        self.claim_sizes = [[] for _ in range(n+1)]

    def get_value_at_time(self, time: float) -> list:
        value = [0 for _ in range(self.n+1)]
        cumm_claim_sizes = []
        # generate claims process until time
        for i in range(0, self.n+1):
            while (self.inter_arrival_times[i] == []) or\
                  (self.inter_arrival_times[i][-1] <= time):
                self._generate_next_claim(i_process=i)
            claim_idx_after_time = self._find_first_claim_idx_after_time_moment(i_process=i, time=time)
            cumm_claim_size = sum(self.claim_sizes[i][:claim_idx_after_time])
            cumm_claim_sizes.append(cumm_claim_size)
        # 1,2,...,n processes
        for i in range(1, self.n+1):
            value[i] = self.u[i] + self.c[i] * time - cumm_claim_sizes[i] + max(cumm_claim_sizes[i] - self.b[i], 0)
        # 0 process
        value[0] = self.u[0] + self.c[0] * time - cumm_claim_sizes[0] - sum([max(cumm_claim_sizes[i] - self.b[i], 0)
                                                                            for i in range(1, self.n+1)])
        return value

    def _generate_next_claim(self, i_process: int):
        self._generate_next_arrival_time(i_process)
        self._generate_next_claim_size(i_process)

    def _generate_next_arrival_time(self, i_process: int):
        if self.inter_arrival_times[i_process]:
            # set seed and generate
            np.random.seed(seed=int(self.seed * (1 + 1e6 * self.inter_arrival_times[i_process][-1]) % 2**32))
            inter_arrival_time = np.random.exponential(1.0 / self.poisson_rate[i_process])
            last_arrival_time = self.inter_arrival_times[i_process][-1] + inter_arrival_time
        else:
            # set seed and generate
            np.random.seed(seed=int(self.seed * (1 + 1e7 * i_process) % 2**32))
            inter_arrival_time = np.random.exponential(1.0 / self.poisson_rate[i_process])
            last_arrival_time = inter_arrival_time
        self.inter_arrival_times[i_process].append(last_arrival_time)

    def _generate_next_claim_size(self, i_process: int):
        claim_size = np.random.exponential(1 / self.claims_rate[i_process])
        self.claim_sizes[i_process].append(claim_size)

    def _find_first_claim_idx_after_time_moment(self, i_process: int, time: float) -> int:
        # using bin search
        left = 0
        right = len(self.inter_arrival_times[i_process])
        while right - left > 1:
            mid = (right + left) // 2
            if self.inter_arrival_times[i_process][mid] <= time:
                left = mid
            else:
                right = mid
        if self.inter_arrival_times[i_process][left] > time:
            return left
        else:
            return right

    def plot_process(self, max_time: float, n_time_points: int = 1000):
        t = np.linspace(0, max_time, n_time_points)
        y = np.empty((self.n+1, n_time_points))
        for time_idx, time in enumerate(t):
            value = self.get_value_at_time(time=time)
            for i, process_value in enumerate(value):
                y[i, time_idx] = process_value
        df = pd.DataFrame(np.vstack([t, y]).T, columns=['time'] + [f'{i} company' for i in range(0, self.n+1)])
        return px.line(df, x='time', y=[f'{i} company' for i in range(0, self.n+1)])
