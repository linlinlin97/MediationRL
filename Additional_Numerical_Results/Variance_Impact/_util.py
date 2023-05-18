from scipy.stats import norm
class normcdf():
    def transform(self, S):
        return norm.cdf(S)

class iden():
    def transform(self, S):
        return S
