import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import pickle
from sklearn.cluster import DBSCAN

class CustomEarlyStopping(tf.keras.callbacks.Callback):
    """ EarlyStopping for "Triplet model". 
        It is designed to monitor 'val_accuracy - max'.
        Training history is available in `history` attribute.
    """

    _accuracy_history = []
    _val_accuracy_history = []
    __train_centroids = None
    __test_centroids = None

    history = None

    def __init__(self, weights_path, accuracy_func, centroids_func, patience=60, batch_size=2048, 
                 verbose=0, history_filepath=None, *args, **kwargs):
        """ weights_path may contain {accuracy}, {val_accuracy}, {epoch}, {loss}, {val_loss} 
            eg. weights_path='weights/01_{val_accuracy:.4f}.h5'
        """
        super().__init__(*args, **kwargs)
        self._weights_path = weights_path
        self._patience = patience
        self._batch_size = batch_size
        self._verbose = verbose
        self.__accuracy_func = accuracy_func
        self.__centroids_func = centroids_func
        self.__best_weights_filepath = None
        self.__history_filepath = history_filepath
        self.__metrics_history = {}
        self.__reset()

    def set_train_data(self, x_train_data, y_train_data):
        self.__x_train_data = x_train_data
        self.__y_train_data = y_train_data

    def set_test_data(self, x_test_data, y_test_data):
        self.__x_test_data = x_test_data
        self.__y_test_data = y_test_data

    def set_train_centroids(self, centroids):
        self.__train_centroids = np.copy(centroids)

    def unset_train_centroids(self):
        self.__train_centroids = None

    def set_test_centroids(self, centroids):
        self.__test_centroids = np.copy(centroids)

    def unset_test_centroids(self):
        self.__test_centroids = None
        
    def __reset(self):
        self._no_improvement = 0
        self._last_acc = 0.0
        self._accuracy_history = []
        self._val_accuracy_history = []
        
    def on_train_begin(self, logs=None):
        self.__reset()

    def on_train_end(self, logs=None):
        self.__generate_history()
        if self.__history_filepath is not None:
            self.save_training_history(self.__history_filepath)
        
    def on_epoch_end(self, epoch, logs=None):
        out_train = self.model.predict(self.__x_train_data, batch_size=self._batch_size)
        out_test = self.model.predict(self.__x_test_data, batch_size=self._batch_size)

        if self.__train_centroids is not None:
            train_centroids = self.__train_centroids
        else:
            train_centroids = self.__centroids_func(out_train, self.__y_train_data)

        if self.__test_centroids is not None:
            test_centroids = self.__test_centroids
        else:
            test_centroids = self.__centroids_func(out_test, self.__y_test_data)
        
        accuracy = self.__accuracy_func(out_train, self.__y_train_data, train_centroids)
        val_accuracy = self.__accuracy_func(out_test, self.__y_test_data, test_centroids)

        self._accuracy_history.append(accuracy)
        self._val_accuracy_history.append(val_accuracy)

        for k in logs.keys():
            if k not in self.__metrics_history.keys():
                self.__metrics_history[k] = []
            self.__metrics_history[k].append(logs[k])

        if self._verbose == 1:
            print(f"[Callback] accuracy: {accuracy} | centroids: {train_centroids.shape[0]} | val_accuracy: {val_accuracy} | val_centroids: {test_centroids.shape[0]}")

        placeholders = {'accuracy' : accuracy, 
                        'val_accuracy' : val_accuracy, 
                        'epoch' : epoch,
                        'loss' : logs['loss'],
                        'val_loss' : logs['val_loss'] if 'val_loss' in logs else None}
         
        if val_accuracy > self._last_acc: 
            self._last_acc = val_accuracy
            self._no_improvement = 0
            self.__best_weights_filepath = self._weights_path.format(**placeholders)
            self.model.save_weights(self.__best_weights_filepath)
        else:
            self._no_improvement += 1

        if self._no_improvement >= self._patience or val_accuracy == 1.0:
            self.model.stop_training=True
            self.model.load_weights(self.__best_weights_filepath)

        if self.__generate_history is not None:
            self.__generate_history()
            self.save_training_history(self.__history_filepath)

    def __generate_history(self):
        self.history = {'accuracy': self._accuracy_history, 'val_accuracy': self._val_accuracy_history, **self.__metrics_history}

    def save_training_history(self, filepath):
        """ Save training history as .pkl 
            (filepath must contain .pkl at the end)
        """
        try:
            with open(filepath, "wb") as f:
                pickle.dump(self.history, f)
        except Exception as e:
            print(e)

def save_training_history(filepath, history):
        """ Save training history as .pkl 
            (filepath must contain .pkl at the end)
        """
        try:
            with open(filepath, "wb") as f:
                pickle.dump(history, f)
        except Exception as e:
            print(e)

def load_training_history(filepath):
    """ Load training history from .pkl

        Returns:
            dict
    """
    try:
        with open(filepath, "rb") as f:
            hist = pickle.load(f)
    except Exception as e:
        print(e)

    return hist

def compute_centroids(embeddings, y_labels):
    """ Compute centroids for each class.
    
    Parameters:
        embeddings: np.array
            - from model.predict(x_train)
        y_labels: np.array
            
    Returns:
        centroids: np.array
    """
    
    c_func = np.mean 
    
    centroids_ids = [x for x in np.unique(y_labels)]
    centroids = np.array([c_func(embeddings[np.argwhere(y_labels==x)[:,0]], axis=0) for x in centroids_ids])
    
    return centroids
    
def compute_centroids_DBSCAN(embeddings, y_labels):
    """ Compute centroids for each class.
    
    Parameters:
        embeddings: np.array
            - from model.predict(x_train)
        y_labels: np.array
            
    Returns:
        centroids: np.array
    """
    
    clustering = DBSCAN(eps=0.1, min_samples=embeddings.shape[0] // 500).fit(embeddings)
    
    centroids_ids = np.unique(clustering.labels_[clustering.labels_ >= 0])
    centroids = np.array([np.mean(embeddings[np.argwhere(clustering.labels_==x)[:,0]], axis=0) for x in centroids_ids])
    
    return centroids


def y_labels_for_regression(y_labels, centroids):
    """ Generate new "y_train" for regression.
    
    Parameters:
        y_labels: np.array 
        centroids: np.array 
            - from compute_centroids()
            
    Returns:
        np.array of shape (samples, embedding_dim,)
    """
    
    return np.array([centroids[x] for x in y_labels])
    
    
def y_labels_for_regression_DBSCAN(embeddings, y_labels, centroids):
    """ Compute centroids for each class.
    
    Parameters:
        embeddings: np.array
            - from model.predict(x_train)
        y_labels: np.array
        min_samples: int
            
    Returns:
        np.array of shape (samples, embedding_dim,)
    """
    
    centroids_labels = np.zeros(centroids.shape[0])
    y_clusters = assign_samples_to_clusters(embeddings, centroids)
    y_clusters_ids = np.unique(y_clusters)
    y_reg_labels = np.zeros(embeddings.shape)
    
    for i in range(centroids.shape[0]):
        centroids_labels[i] = np.argmax(np.bincount(y_labels[np.argwhere(y_clusters == y_clusters_ids[i])[:,0]]))
    
    for i in range(embeddings.shape[0]):
        distances = np.array([np.linalg.norm(embeddings[i] - centroids[j]) if y_labels[i] == centroids_labels[j] else float('inf') for j in range(centroids_labels.shape[0])])
        y_reg_labels[i] = centroids[np.argmin(distances)]

    return y_reg_labels
    

def assign_samples_to_clusters(y_pred, centroids, ord=None):
    """ Assign sample to its nearest cluster.
    
    Parameters:
        y_pred: np.array 
            - from model.predict(x)
        centroids: np.array
            - from compute_centroids()
        ord:{non-zero int, inf, -inf, ‘fro’, ‘nuc’}, default None
            - 1 = L1, 2 = L2, ...
            - https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html

    Returns:
        nearest_centroids: np.array
    """
    
    distances = [np.linalg.norm(y_pred[x] - centroids, axis=1, ord=ord) for x in range(y_pred.shape[0])]
    return np.argmin(distances, axis=1).astype(np.uint32)

def compute_accuracy(y_pred, y_true, centroids, ord=None):
    """ Compute accuracy - how often predictions equals labels.
    
    Parameters:
        y_pred: np.array
            - from model.predict(x)
        y_true: np.array
            - ground truth values
        centroids: np.array
            - from compute_centroids()
        ord:{non-zero int, inf, -inf, ‘fro’, ‘nuc’}, default None
            - 1 = L1, 2 = L2, ...
            - https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html

    Returns:
        accuracy
    """
    
    y_pred_clusters = assign_samples_to_clusters(y_pred=y_pred,
                                                centroids=centroids,
                                                ord=ord)
    
    acc = tf.keras.metrics.Accuracy()
    acc.update_state(y_pred=y_pred_clusters, y_true=y_true)
    return acc.result().numpy()
    
def compute_accuracy_DBSCAN(y_pred, y_true, centroids, ord=None):
    """ Compute accuracy - how often predictions equals labels.
    
    Parameters:
        y_pred: np.array
            - from model.predict(x)
        y_true: np.array
            - ground truth values
        centroids: np.array
            - from compute_centroids()
        ord:{non-zero int, inf, -inf, ‘fro’, ‘nuc’}, default None
            - 1 = L1, 2 = L2, ...
            - https://numpy.org/doc/stable/reference/generated/numpy.linalg.norm.html

    Returns:
        accuracy
    """
    
    if centroids.shape[0] == 0:
        return 0

    y_pred_clusters = assign_samples_to_clusters(y_pred=y_pred, centroids=centroids, ord=ord)
    y_pred_ids = np.unique(y_pred_clusters)
    accuracies = np.zeros(y_pred_ids.shape[0])

    for i in range(y_pred_ids.shape[0]):
        accuracies[i] = np.max(np.bincount(y_true[np.argwhere(y_pred_clusters == y_pred_ids[i])[:,0]])) / np.argwhere(y_pred_clusters == y_pred_ids[i]).shape[0]

    return np.average(accuracies)

def assign_samples_to_clusters_cos_similarity(y_pred, centroids):
    dot = np.matmul(y_pred, centroids.T)
    norms = np.array([np.linalg.norm(y_pred, axis=1) * np.linalg.norm(c) for c in centroids]).T
    cossims = dot / (norms + 1e-12)
    return np.argmax(cossims, axis=1).astype(np.uint32)


def compute_cos_accuracy(y_pred, y_true, centroids):
    y_pred_clusters = assign_samples_to_clusters_cos_similarity(y_pred=y_pred,
                                                                centroids=centroids)
    
    acc = tf.keras.metrics.Accuracy()
    acc.update_state(y_pred=y_pred_clusters, y_true=y_true)
    return acc.result().numpy()


def load_reference_model(architecture_filepath,
                         weights_filepath=None):
    """ Load and compile reference model
            - optimizer: Adam, 
            - loss: SparseCategoricalCrossentropy
            - metrics: Accuracy 

        weights_filepath: filepath or None
    """

    with open(architecture_filepath, "r") as f:
        model = tf.keras.models.model_from_json(f.read())

    if weights_filepath is not None:
        model.load_weights(weights_filepath)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                 loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                 metrics=['accuracy'])
    
    return model

def load_triplet_model(architecture_filepath,
                       weights_filepath=None, 
                       margin=1.0):
    """ Load and compile triplet model
        - optimizer: Adam,
        - loss: TripletSemiHardLoss(margin)

        weights_filepath: filepath or None
    """

    margin = float(margin)
    
    with open(architecture_filepath, "r") as f:
        model = tf.keras.models.model_from_json(f.read())

    if weights_filepath is not None:
        model.load_weights(weights_filepath)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                 loss=tfa.losses.TripletSemiHardLoss(margin=margin))
    
    return model


def normalize_centroids(centroids, a=0, b=1):
    """ normalize centroids to <a,b> """
    centroids = np.copy(centroids)

    centroids = (centroids - np.min(centroids, axis=0)) / (np.max(centroids, axis=0) - np.min(centroids, axis=0))
    centroids = centroids * (b-a)
    centroids = centroids + a
    return centroids

def minmax_centroids(centroids):
    """ set min = 0, max = 1 for each dimension """
    c = np.copy(centroids)
    mmax = np.argmax(c, axis=0)
    mmin = np.argmin(c, axis=0)
    
    for x in range(c.shape[1]):
        c[:, x][mmax[x]] = 1
        c[:, x][mmin[x]] = 0
        
    return c

def y_labels__A(labels):

    y_train_A = np.zeros((labels.shape[0], 4))

    new_labels_A = np.array([
                        [0,1,1,1],
                        [0,0,1,1],
                        [1,1,0,1],
                        [0,0,0,1],
                        [0,0,0,0],
                        [0,0,1,0],
                        [0,1,0,0],
                        [1,0,1,1],
                        [1,0,1,0],
                        [1,1,1,1]])  

    for i in range(10):
        y_train_A[labels==i] = new_labels_A[i]

    return y_train_A

def y_labels__B(labels):

    y_train_B = np.zeros((labels.shape[0], 4))

    new_labels_B = np.array([
                        [1,0,1,0],
                        [0,1,1,0],
                        [1,1,1,1],
                        [0,1,1,1],
                        [1,1,1,0],
                        [0,1,0,0],
                        [0,0,0,1],
                        [1,1,0,0],
                        [1,0,0,1],
                        [1,0,1,1]])  

    for i in range(10):
        y_train_B[labels==i] = new_labels_B[i]

    return y_train_B

def generate_random_codes():
    """ Generate random 4D codes """
    a = np.random.choice(range(16), 10, replace=False)
    a = np.expand_dims(a, axis=1).astype(np.uint8)
    codes = np.unpackbits(a,axis=1)[:, 4:]
    return codes

def save_codes_as_txt(filepath, codes, fmt='%.0f'):
    np.savetxt(filepath, codes, fmt=fmt, delimiter=',')

def cifar10_y_labels__c_01(labels):
    y_new = np.zeros((labels.shape[0], 4))

    y = np.array([
            [0,1,1,0],
            [0,0,1,1],
            [0,1,0,1],
            [1,0,0,1],
            [0,1,1,1],
            [1,1,0,0],
            [0,0,0,1],
            [1,1,1,0],
            [0,0,0,0],
            [1,0,1,0]])

    for i in range(10):
        y_new[labels==i] = y[i]

    return y_new

def cifar10_y_labels__c_02(labels):
    y_new = np.zeros((labels.shape[0], 4))

    y = np.array([
            [1,1,1,1],
            [0,1,1,0],
            [1,0,0,1],
            [0,0,1,0],
            [0,0,0,1],
            [1,0,1,0],
            [1,1,0,0],
            [0,0,1,1],
            [1,1,1,0],
            [0,1,1,1],
            ])

    for i in range(10):
        y_new[labels==i] = y[i]

    return y_new

def cifar10_y_labels__c_03(labels):
    y_new = np.zeros((labels.shape[0], 4))

    y = np.array([
            [0,0,0,0],
            [0,0,0,1],
            [1,0,1,0],
            [1,1,0,1],
            [0,0,1,1],
            [1,1,1,1],
            [1,0,1,1],
            [0,1,1,0],
            [1,0,0,0],
            [0,1,0,1]])

    for i in range(10):
        y_new[labels==i] = y[i]

    return y_new

