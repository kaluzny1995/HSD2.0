class Classifier:
    def __init__(self, name='Classifier'):
        self.name = name
        self._fit = False
    
    def __str__(self):
        pass
    
    def fit(self, X, y):
        self._fit = True
        pass
    
    def predict(self, X):
        if not self._fit:
            raise ValueError(f'Classifier: {self.name} must be fit first!')
        pass

    def test(self, text):
        if not self._fit:
            raise ValueError(f'Classifier: {self.name} must be fit first!')
        pass

    def save(self, save_file):
        if not self._fit:
            raise ValueError(f'Classifier: {self.name} must be fit first!')
        pass

    def load(self, load_file):
        self._fit = True
        pass
