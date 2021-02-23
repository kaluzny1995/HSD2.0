class Vectorizer:
    def __init__(self, name='Vectorizer'):
        self.name = name
        self._fit = False
    
    def __str__(self):
        return self.name
    
    def fit(self):
        self._fit = True
        pass
    
    def transform(self):
        if not self._fit:
            raise ValueError(f'Vectorizer: {self.name} must be fit first!')
        pass
    
    def fit_transform(self):
        self._fit = True
        pass
