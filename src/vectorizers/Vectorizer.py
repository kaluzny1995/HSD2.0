class Vectorizer:
    def __init__(self, name='Vectorizer'):
        self.name = name
        self._fit = False
    
    def __str__(self):
        return self.name
    
    def fit(self, X):
        self._fit = True
        pass
    
    def transform(self, X):
        if not self._fit:
            raise ValueError(f'Vectorizer: {self.name} must be fit first!')
        pass
    
    def fit_transform(self, X):
        self._fit = True
        pass

    def save(self, save_file):
        if not self._fit:
            raise ValueError(f'Vectorizer: {self.name} must be fit first!')
        pass

    def load(self, load_file):
        self._fit = True
        pass
