import pyprob

class Simulator(pyprob.Model):
    """ Abstract simulator interface. """
    # def __init__(**kwargs):
    #     super().__init__(kwargs)

    def forward(self, inputs):
        raise NotImplementedError()

    def log_prob(self, inputs, outputs):
        raise IntractableException

    def __call__(self, inputs, **kwargs):
        return self.forward(inputs, **kwargs)


class IntractableException(Exception):
    pass
