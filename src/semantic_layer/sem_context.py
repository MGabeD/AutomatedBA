from src.step import Step
from abc import ABC, abstractmethod
from src.semantic_layer import BaseSemantics

class SemanticContext(BaseSemantics):
    def __init__(self, args=None):
        super().__init__(args=args)
        

    def run(self, args):
        args["semantics_context"] = self.schema
        return args

def main():
    a = SemanticContext()
    b = a.schema
    c = a.run({})
    print(c)

if __name__ == "__main__":
    main()