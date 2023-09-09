
class Base:
    pass


class PatterA(Base):
    pass


class View:
    def __init__(self, inp: PatterA):
        print(isinstance(inp, PatterA))

if __name__ == '__main__':
    View(inp=PatterA())