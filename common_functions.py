from functools import partial
import numpy as np
from scipy.integrate import quad


def pdf_exp(lambd: float, x: float):
    return lambd * np.exp(- lambd * x)


def int_hat_pdf_exp_function(s: float, lambd: float, x: float):
    return np.exp(- s * x) * pdf_exp(lambd=lambd, x=x)


def calculate_hat_pdf_exp(s: float, lambd: float, lower_limit: float = 0, upper_limit: float = np.inf) -> float:
    int_hat_pdf_exp_function_instance = partial(
        int_hat_pdf_exp_function,
        s,
        lambd
    )
    return quad(int_hat_pdf_exp_function_instance, lower_limit, upper_limit)[0]
