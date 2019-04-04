import math

#https://math.stackexchange.com/questions/1689581/inverse-standard-normal-cdf

def pade_approx_norminv(p):
    q = math.sqrt(2*math.pi) * (p - 1/2) - (157/231) * math.sqrt(2) * math.pi**(3/2) * (p - 1/2)**3
    r = 1 - (78/77) * math.pi * (p - 1/2)**2 + (241* math.pi**2 / 2310) * (p - 1/2)**4
    return q/r

def d_prime(hit_rate, false_alarm_rate):
    return pade_approx_norminv(hit_rate) - pade_approx_norminv(false_alarm_rate)