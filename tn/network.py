from keras.utils.np_utils import to_categorical
from keras import models
from keras import layers
from keras.models import load_model
import numpy as np
import os
from sklearn.feature_extraction.text import CountVectorizer
from pandas import read_csv, DataFrame

class TweetAnalysisNetwork:

    def __init__(self, data_path: str, data_rows: int = None, name: str = None):
        """ Initialize network
        Args:
            name (str): Model name
        """

        if not data_rows:
            data_rows = 15000

        df = read_csv(data_path, nrows=data_rows)

        self.features = df.text.array
        self.labels = df.label.array
        self.name = name if name else 'model.h5'
        self.vectorizer = CountVectorizer()
        self.vectorizer.fit(self.features)
        self.size = len(self.vectorizer.vocabulary_)


    def create(self, optimizer: str = None,
            loss: str = None, metrics: list = None,
            epochs: int = None, batch_size: int = None):

        """Create the new network
        Args:
            optimizer (str): Network optimizer
            loss (str): Network loss function
            epochs (int): Network epochs number
            metrics (list): Network matrics list
            batch_size (int): Network batch size
        Return:
            models.Sequential - loaded sequential model
        """
        # Set the default values of parapms if them won`t set

        self.optimizer = optimizer if optimizer else 'rmsprop'
        self.loss = loss if loss else 'categorical_crossentropy'
        self.metrics = metrics if metrics else ['accuracy']
        self.batch_size = batch_size if batch_size else 512
        self.epochs = epochs if epochs else 7

        # Define network layers
        self.__first_layer =  layers.Dense(64,
                activation='relu', input_shape=(self.size,))
        self.__second_layer = layers.Dense(64, activation='relu')
        self.__last_layer = layers.Dense(3, activation='softmax')

        # Construct our networkoptimizer
        self.model = models.Sequential()
        self.model.add(self.__first_layer)
        self.model.add(self.__second_layer)
        self.model.add(self.__last_layer)

        return self.model


    def compile(self):
        """Compole network for future usage
        Return:
            None
        """
        self.model.compile(optimizer=self.optimizer,
                loss=self.loss, metrics=self.metrics)


    def train(self, train_size: int = None):
        """Train neural network on given data
        Args:
            train_features (list): List of twitter text
            train_labels (list): List of twitter categories (0-2)
        Return:
            None
        """
        train_size = train_size if train_size else 10000

        train_features = self.vectorize(self.features[:train_size])
        train_labels = self.categorize(self.labels[:train_size])

        self.model.fit(train_features, train_labels,
                epochs=self.epochs, batch_size=self.batch_size)


    def save(self, path: str):
        """Save trained model in given path
        Args:
            path (str): Path to save the model
        """
        if not os.path.exists(path):
            os.makedirs(path)
        self.model.save(f'{path}/{self.name}.h5')


    def load(self, path: str):
        """Load saved model by given path
        Args:
            path (str): The path where the model was saved
        Return:
            models.Sequential - loaded sequential model
        """
        self.model = load_model(path)
        return self.model


    def predict(self, text: str, json = None):
        """Predict information via loaded model
        Args:
            text (str): The text to analize
            json (bool): If true then return pretty printed json
        Return:
            list - list of predictions
        """
        if not json:
            json = False
        prediction = self.model.predict(self.vectorize([text]))[0]
        if json:
            return {
                'neutral': {
                    'probability': prediction[2]*100,
                    'name': 'Нейтральный'
                },
                'positive': {
                    'probability': prediction[1]*100,
                    'name': 'Положительный'
                },
                'negative': {
                    'probability': prediction[0]*100,
                    'name': 'Отрицательный'
                }
            }
        else:
            return prediction


    def vectorize(self, text_list: list):
        return self.vectorizer.transform(text_list).todense()


    def categorize(self, labels: list):
        return to_categorical(np.array(labels), num_classes=3)
