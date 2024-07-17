class UnknownArgs(ValueError):
    def __init__(self, message):
        self.msg = message
        super().__init__(self.msg)

    
    