"""Scale features to the unit interval."""

from sklearn.preprocessing import MinMaxScaler


def example_data():
    X1 = [[1, 2],
         [2, 1],
         [3, 2],
         [4, 3]]
    X2 = [[1.5, 2864, 2.3],
          [2.6, 8372, 1.8],
          [1.2, 6453, 2.2],
          [2.3, 9587, 3.7],
          [1.9, 2332, 3.1],
          [3.7, 8574, 1.5],
          [2.1, 7665, 2.3],
          [1.4, 2428, 1.8],
          [3.7, 9476, 3.2],
          [1.5, 3422, 2.4]]
    return X1, X2


def normalize(features):
    scaler = MinMaxScaler()
    return scaler.fit_transform(features), scaler


def main():
    for features in example_data():
        normalized, _ = normalize(features)
        print(normalized)

if __name__ == "__main__":
    main()
