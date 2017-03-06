from sklearn.model_selection import train_test_split
from ryan.preprocess import preprocess
from ryan.logistic_regression import LogisticRegression

SEED = 1337


def main():
    x_matrix, y_matrix = preprocess()
    # Split data into testing and training sets
    x_train, x_test, y_train, y_test = train_test_split(
        x_matrix, y_matrix, test_size=0.2, random_state=SEED
    )
    regression = LogisticRegression(x_train, y_train, num_epochs=1000000, seed=SEED)
    print("Test Accuracy: %f" % regression.test_accuracy(x_test, y_test))
    regression.save_model()

if __name__ == "__main__":
    main()