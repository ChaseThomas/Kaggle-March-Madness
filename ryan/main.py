import tensorflow as tf
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

SEED = 1337
FILEPATH = "data/processed.csv"
LEARNING_RATE = 0.05
NUM_EPOCHS = 1000
BATCH_SIZE = 100


def shooting(fgm, fga, fgm3):
    if fga == 0:
        return 0
    else:
        return (fgm + 0.5 * fgm3) / fga


def turnovers(to, fga, fta):
    tmp = fga + 0.44 * fta + to
    if tmp == 0:
        return 0
    else:
        return to / tmp


def off_rebounds(my_or, opp_dr):
    tmp = my_or + opp_dr
    if tmp == 0:
        return 0
    else:
        return my_or / tmp


def def_rebounds(my_dr, opp_or):
    tmp = my_dr + opp_or
    if tmp == 0:
        return 0
    else:
        return my_dr / tmp


def freethrows(ftm, fta):
    if fta == 0:
        return 0
    else:
        return ftm / fta


def process_row(x):
    w_shooting = shooting(x['Wfgm'], x['Wfga'], x['Wfgm3'])
    l_shooting = shooting(x['Lfgm'], x['Lfga'], x['Lfgm3'])
    w_turnovers = turnovers(x['Wto'], x['Wfga'], x['Wfta'])
    l_turnovers = turnovers(x['Lto'], x['Lfga'], x['Lfta'])
    w_off_rebounds = off_rebounds(x['Wor'], x['Ldr'])
    l_off_rebounds = off_rebounds(x['Lor'], x['Wdr'])
    w_def_rebounds = def_rebounds(x['Wdr'], x['Lor'])
    l_def_rebounds = def_rebounds(x['Ldr'], x['Wor'])
    w_freethrows = freethrows(x['Wftm'], x['Wfta'])
    l_freethrows = freethrows(x['Lftm'], x['Lfta'])
    result = {
        'def_rebounds': w_def_rebounds - l_def_rebounds,
        'freethrows': w_freethrows - l_freethrows,
        'off_rebounds': w_off_rebounds - l_off_rebounds,
        'shooting': w_shooting - l_shooting,
        'turnovers': w_turnovers - l_turnovers,
    }
    return pd.Series(result)


def main():
    processed_file = Path(FILEPATH)
    if processed_file.is_file():
        print("Loading preprocessed data!")
        processed_df = pd.read_csv(processed_file, dtype=np.float32, skipinitialspace=True)
        print("Finished loading data!")
    else:
        print("Calculating data for preprocessing!")
        df_full = pd.read_csv(
            "data/RegularSeasonDetailedResults.csv",
            usecols=('Wfgm',  'Lfgm',  'Wfga',  'Lfga',  'Wfgm3', 'Lfgm3', 'Wto',   'Lto',
                     'Wfta',  'Lfta',  'Wor',   'Lor',   'Wdr',   'Ldr',   'Wftm',  'Lftm'),
            dtype=np.float32,
            skipinitialspace=True)
        processed_df = df_full.apply(process_row, axis=1)
        processed_df.to_csv(FILEPATH)
        print("Finished calculating data for preprocessing!")

    # Setup x_matrix
    x_matrix = processed_df.values[:, 1:]  # Strip index column and return as numpy matrix
    x_matrix = np.repeat(x_matrix, 2, axis=0)  # Duplicate each row
    x_matrix[1::2, :] *= -1  # Invert every other row to represent losers

    # Setup y_matrix
    y_matrix = np.empty(x_matrix.shape[0], dtype=np.float32)

    y_matrix[::2] = 1  # Winners
    y_matrix[1::2] = 0  # Losers

    # SETUP THE MODEL
    # Input
    x_data = tf.constant(processed_df.as_matrix(), dtype=tf.float32, name="Input", shape=processed_df.shape)
    y_data = tf.constant(y_matrix, dtype=tf.float32, name="Output")

    # Symbolic Vars
    X = tf.placeholder(tf.float32, (None, 5), name="Features")
    Y = tf.placeholder(tf.bool, (None, 1), name="Targets")

    # Parameters
    w = tf.Variable(tf.random_normal((5, 1), stddev=0.01, name="Parameters"))
    b = tf.Variable(tf.random_normal((1,), stddev=0.01, name="Bias"))

    # Output Function
    y = tf.matmul(X, w) + b

    # Loss Function
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y),
        name='Loss'
    )
    # Gradient Descent and Output Calculation
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
    prediction_step = tf.argmax(y, 1)

    # Training Loop
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(NUM_EPOCHS):
            epoch_loss = 0
            for _ in range(int(processed_df.shape[0] / BATCH_SIZE)):
                pass
                # x_batch, y_batch = x_data.n
                # TODO: Finish the training loop


if __name__ == "__main__":
    main()
