import tempfile
import tensorflow as tf

from six.moves import urllib

import pandas as pd

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string("model_dir","","Base directory for output models.")
flags.DEFINE_string("model_type","wide_n_deep","valid model types:{'wide','deep', 'wide_n_deep'")
flags.DEFINE_integer("train_steps",200,"Number of training steps.")
flags.DEFINE_string("train_data","", "Path to the training data.")
flags.DEFINE_string("test_data", "", "path to the test data")

COLUMNS = ["age", "workclass", "fnlwgt", "education", "education_num",
           "marital_status", "occupation", "relationship", "race", "gender",
           "capital_gain", "capital_loss", "hours_per_week", "native_country",
           "income_bracket"]

LABEL_COLUMN = "label"

CATEGORICAL_COLUMNS = ["workclass", "education", "marital_status", "occupation",
                       "relationship", "race", "gender", "native_country"]

CONTINUOUS_COLUMNS = ["age", "education_num", "capital_gain", "capital_loss",
                      "hours_per_week"]

# download test and train data
def maybe_download():
    if FLAGS.train_data:
        train_data_file = FLAGS.train_data
    else:
        train_file = tempfile.NamedTemporaryFile(delete=False)
        urllib.request.urlretrieve("http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.data", train_file.name)
        train_file_name = train_file.name
        train_file.close()
        print("Training data is downloaded to %s" % train_file_name)

    if FLAGS.test_data:
        test_file_name = FLAGS.test_data
    else:
        test_file = tempfile.NamedTemporaryFile(delete=False)
        urllib.request.urlretrieve("http://mlr.cs.umass.edu/ml/machine-learning-databases/adult/adult.test",
                                   test_file.name)  # pylint: disable=line-too-long
        test_file_name = test_file.name
        test_file.close()
        print("Test data is downloaded to %s" % test_file_name)

    return train_file_name, test_file_name

# build the estimator
def build_estimator(model_dir):

    gender = tf.contrib.layers.sparse_column_with_keys(column_name="gender", keys=["female","male"])
    education = tf.contrib.layers.sparse_column_with_hash_bucket("education", hash_bucket_size = 1000)
    relationship = tf.contrib.layers.sparse_column_with_hash_bucket("relationship", hash_bucket_size = 100)
    workclass = tf.contrib.layers.sparse_column_with_hash_bucket("workclass", hash_bucket_size=100)
    occupation = tf.contrib.layers.sparse_column_with_hash_bucket("occupation", hash_bucket_size=1000)
    native_country = tf.contrib.layers.sparse_column_with_hash_bucket( "native_country", hash_bucket_size=1000)

    # Continuous base columns.
    age = tf.contrib.layers.real_valued_column("age")
    education_num = tf.contrib.layers.real_valued_column("education_num")
    capital_gain = tf.contrib.layers.real_valued_column("capital_gain")
    capital_loss = tf.contrib.layers.real_valued_column("capital_loss")
    hours_per_week = tf.contrib.layers.real_valued_column("hours_per_week")

    age_buckets = tf.contrib.layers.bucketized_column(age, boundaries= [18,25, 30, 35, 40, 45, 50, 55, 60, 65])

    wide_columns = [gender, native_country,education, occupation, workclass, relationship, age_buckets,
                    tf.contrib.layers.crossed_column([education, occupation], hash_bucket_size=int(1e4)),
                    tf.contrib.layers.crossed_column([age_buckets, education, occupation], hash_bucket_size=int(1e6)),
                    tf.contrib.layers.crossed_column([native_country, occupation],hash_bucket_size=int(1e4))]


    deep_columns = [tf.contrib.layers.embedding_column(workclass, dimension=8),
                    tf.contrib.layers.embedding_column(education, dimension=8),
                    tf.contrib.layers.embedding_column(gender, dimension=8),
                    tf.contrib.layers.embedding_column(relationship, dimension=8),
                    tf.contrib.layers.embedding_column(native_country,dimension=8),
                    tf.contrib.layers.embedding_column(occupation, dimension=8),
                    age,education_num,capital_gain,capital_loss,hours_per_week,]

    if FLAGS.model_type =="wide":
        m = tf.contrib.learn.LinearClassifier(model_dir=model_dir,feature_columns=wide_columns)
    elif FLAGS.model_type == "deep":
        m = tf.contrib.learn.DNNClassifier(model_dir=model_dir, feature_columns=deep_columns, hidden_units=[100,50])
    else:
        m = tf.contrib.learn.DNNLinearCombinedClassifier(model_dir=model_dir, linear_feature_columns=wide_columns, dnn_feature_columns = deep_columns, dnn_hidden_units=[100,50])

    return m

def input_fn(df):
    continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
    categorical_cols = {k: tf.SparseTensor(indices=[[i,0] for i in range( df[k].size)], values = df[k].values, dense_shape=[df[k].size,1]) for k in CATEGORICAL_COLUMNS}
    feature_cols = dict(continuous_cols)
    feature_cols.update(categorical_cols)
    label = tf.constant(df[LABEL_COLUMN].values)

    return feature_cols, label


def train_and_eval():
    train_file_name, test_file_name = maybe_download()
    df_train = pd.read_csv(
        tf.gfile.Open(train_file_name),
        names=COLUMNS,
        skipinitialspace=True,
        engine="python"
    )
    df_test = pd.read_csv(
        tf.gfile.Open(test_file_name),
        names=COLUMNS,
        skipinitialspace=True,
        skiprows=1,
        engine="python"
    )

    # drop Not a number elements
    df_train = df_train.dropna(how='any',axis=0)
    df_test = df_test.dropna(how='any', axis=0)

    #convert >50 to 1
    df_train[LABEL_COLUMN] = (
        df_train["income_bracket"].apply(lambda x: ">50" in x).astype(int)
    )
    df_test[LABEL_COLUMN] = (
        df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)

    model_dir = tempfile.mkdtemp() if not FLAGS.model_dir else FLAGS.model_dir
    print("model dir = %s" % model_dir)

    m = build_estimator(model_dir)
    print (FLAGS.train_steps)
    m.fit(input_fn=lambda: input_fn(df_train),
          steps=FLAGS.train_steps)
    results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)

    for key in sorted(results):
        print("%s: %s"%(key, results[key]))

def main(_):
  train_and_eval()


if __name__ == "__main__":
  tf.app.run()