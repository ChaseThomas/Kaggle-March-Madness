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
        (fgm+0.5*fgm3) / fga


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
    result = {
        'shooting': shooting(x['Wfgm']-x['Lfgm'], x['Wfga']-x['Lfga'], x['Wfgm3']-x['Lfgm3']),
        #'Lshooting': shooting(x['Lfgm'], x['Lfga'], x['Lfgm3']),
        'Wturnovers': turnovers(x['Wto']-x['Lto'], x['Wfga']-x['Lfga'], x['Wfta']-x['Lfta']),
        #'Lturnovers': turnovers(x['Lto'], x['Lfga'], x['Lfta']),
        'Woff_rebounds': off_rebounds(x['Wor']-x['Lor'], x['Ldr']-x['Wdr']),
        #'Loff_rebounds': off_rebounds(x['Lor'], x['Wdr']),
        'Wdef_rebounds': def_rebounds(x['Wdr']-x['Ldr'], x['Lor']-x['Wor']),
        #'Ldef_rebounds': def_rebounds(x['Ldr'], x['Wor']),
        'Wfreethrows': freethrows(x['Wftm']-x['Lftm'], x['Wfta']-x['Lfta']),
        #'Lfreethrows': freethrows(x['Lftm'], x['Lfta']),
    }
    return pd.Series(result)

def randomize_row(x):
    pass


def main():
    processed_file = Path(FILEPATH)
    if processed_file.is_file():
        print("Loading preprocessed data!")
        processed_df = pd.read_csv(processed_file, skipinitialspace=True)
        print("Finished loading data!")
    else:
        print("Calculating data for preprocessing!")
        df_full = pd.read_csv("data/RegularSeasonDetailedResults.csv", skipinitialspace=True)
        processed_df = df_full.apply(process_row, axis=1)
        processed_df.to_csv(FILEPATH)
        print("Finished calculating data for preprocessing!")
    x_matrix = processed_df.values
    #print(x_matrix)
    np.apply_along_axis(randomize_row, axis=1, arr=x_matrix)
    #TODO: Fix this shit



    #SETUP THE MODEL
    #Input
    x_data = tf.constant(processed_df.as_matrix(), dtype=tf.float32, name="Input", shape=processed_df.shape)


    #Symbolic Vars
    X = tf.placeholder(tf.float32, (None, 5), name="Features")
    Y = tf.placeholder(tf.bool, (None, 1), name="Target")

    #Parameters
    w = tf.Variable(tf.random_normal((5, 1), stddev=0.1, name="Parameters"))
    b = tf.Variable(tf.random_normal((1,), stddev=0.1, name="Bias"))

    #Output Function
    y = tf.matmul(X, w) + b

    #Loss Function
    loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=y)
    )

    #Gradient Descent and Output Calculation
    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
    prediction_step = tf.argmax(y, 1)

    #Training Loop
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(NUM_EPOCHS):
            epoch_loss = 0
            for _ in range(int(processed_df.shape[0]/BATCH_SIZE)):
                pass
                #x_batch, y_batch = x_data.n
                #TODO: Finish the training loop

if __name__ == "__main__": main()

