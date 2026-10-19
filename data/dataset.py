from sklearn.model_selection import train_test_split

class Dataset:

    def __init__(self, X, y, test_size=0.2, shuffle=True):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, shuffle=shuffle
        )

    def get_train_data(self):
        return self.X_train, self.y_train
    
    def get_test_data(self):
            return self.X_test, self.y_test
