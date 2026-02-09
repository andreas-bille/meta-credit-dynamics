from dataclasses import dataclass

@dataclass
class EWMAStats:
    beta: float
    mu: float = 0.0
    var: float = 1.0
    dd: float = 0.0
    cum_pi: float = 0.0
    peak_cum_pi: float = 0.0
    seed_var: float = 1.0

    def update(self, pi: float):
        mu_new = (1 - self.beta) * self.mu + self.beta * pi
        var_new = (1 - self.beta) * self.var + self.beta * (pi - self.mu) ** 2
        self.mu = mu_new
        self.var = max(var_new, 0.0)

        # Drawdown on cumulative net flow
        self.cum_pi += pi
        if self.cum_pi > self.peak_cum_pi:
            self.peak_cum_pi = self.cum_pi
        self.dd = self.peak_cum_pi - self.cum_pi

    def reset(self):
        self.mu = 0.0
        self.var = self.seed_var
        self.dd = 0.0
        self.cum_pi = 0.0
        self.peak_cum_pi = 0.0
