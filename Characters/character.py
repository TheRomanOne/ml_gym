import utils


__MOBILITY__ = {
        'SOLID': {
            'linear': {'min': [0, 0, 0], 'max': [0, 0, 0]},
            'angular': {'min': [0, 0, 0], 'max': [0, 0, 0]}
            },

        'LOOSE': {
            'linear': {'min': [0, 0, -1], 'max': [0, 0, 1]},
            'angular': {'min': [-0.2, -0.2, 0], 'max': [0.2, 0.2, 0]}
            }
}

class Character:
    def __init__(self, render, loader, world) -> None:
        self.render = render
        self.world = world
        self.loader = loader
        self.brain = None
        self.objects = []
        self.models = []

    
    
    def assign_target(self, target):
        self.target = target
    
    def remove(self):
        self.target.removeNode()
        del self.target

    def interact(self):
        pass

    def evaluate(self):
        pass
    