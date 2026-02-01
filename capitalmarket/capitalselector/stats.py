from dataclasses import dataclass

@dataclass
class EWMAStats:
    beta: float
    mu: float = 0.0
    var: float = 1.0

    def update(self, r: float):
        mu_new = (1 - self.beta) * self.mu + self.beta * r
        var_new = (1 - self.beta) * self.var + self.beta * (r - self.mu) ** 2
        self.mu = mu_new
        self.var = max(var_new, 0.0)