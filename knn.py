import numpy as np
from collections import Counter
import sklearn.metrics.pairwise


def predict(X_train, y_train, X_test, k):
    distances = []
    topcategories = []
    # For every document from the train set
    for i in range(X_train.shape[0]):
        # Calculate the cosine distance from the current test document
        distance = sklearn.metrics.pairwise.cosine_distances(X_test, X_train[i])
        print("\n\n",distance,"\n\n\n")
        distance = distance[0][0]
        # Add the distance to a list along with the documents index
        distances.append([distance, i])
    # Sort the list
    print("\n\n\n",distances,"\n\n\n\n")
    distances = sorted(distances)
    print("\n\n\n",distances,"\n\n\n\n")
    # For the top K elements of the list
    for i in range(k):
        index = distances[i][1]
        print("\n\n\n",index,"\n\n\n yoooo")
        topcategories.append(y_train[index])
        print("\n\n\n\n top",y_train[index],"\n\n\n\n label")
    # Find the most common category
    commoncategory = Counter(topcategories).most_common(1)[0][0]
    return commoncategory


def kNearestNeighbor(X_train, y_train, X_test, k):
    predictions = []
    # For every document from the test set
    for i in range(X_test.shape[0]):
        predictions.append(predict(X_train, y_train, X_test[i], k))
    predictions = np.asarray(predictions)
    return predictions
