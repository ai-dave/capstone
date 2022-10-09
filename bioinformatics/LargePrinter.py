class LargePrinter:
    def __init__(self):
        self.print_q = []
        
    def set_list(self, list):
        self.print_q = list;
        
    def append(self, statement):
        self.print_q.append(statement)
        
    def print(self):
        i = 1
        for statement in self.print_q:
            self.print_one(statement, 0)
            
    def print_one(self, statement, i=0):
        if i > 0:
            print(f'{i}\t', end = "")
        elif i < 0:
            print(f'...\t', end = "")
        print(f'{statement}')
            
    def head_tail(self, size=10):
        i = 1
        break_printed = False
        for statement in self.print_q:
            if i<=size or i>len(self.print_q)-size:
                self.print_one(statement, i)
            else:
                if not break_printed:
                    self.print_one('...', -i)
                    break_printed = True
            i += 1
            
    def print_list(self, list):
        self.set_list(list)
        self.print()
        
    def head_tail_list(self, list, size=10):
        self.size = size
        self.set_list(list)
        self.head_tail(size=self.size)
        