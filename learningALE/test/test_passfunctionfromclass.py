class foo:
    def __init__(self):
        self.bar = 0

    def set_bar(self):
        self.bar = 1

    def print_bar(self):
        print(self.bar)

a = foo()

foo.set_bar(a)

a.print_bar()