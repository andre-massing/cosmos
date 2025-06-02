class Scheme():

    def __init__(self):
        raise Exception('Base class method is called! Empty method!')

class Steady(Scheme):

    def __init__(self):
        self.order = 0

class BDF1(Scheme):

    def __init__(self):
        self.order = 1

class BDF2(Scheme):

    def __init__(self):
        self.order = 2

class CN(Scheme):

    def __init__(self):
        self.order = 2