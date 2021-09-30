class NoSchedule:
    def __init__(self, lr, **kwargs):
        self.lr = lr

    def __call__(self, *args, **kwargs):
        return self.lr, self.lr
