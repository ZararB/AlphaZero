class Node(object):


    def __init__(self, prior : float):
        self.visit_count = 0
        self.to_play = -1
        self.prior = prior #TODO Understand what prior is
        self.value_sum = 0 
        self.children = {} 

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count


