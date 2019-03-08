import flask
from flask import request, jsonify
import sys
import logging
from googleapiclient import discovery
from createFeatureSets import CreateFeatureSet
import tensorflow as tf
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

app = flask.Flask(__name__)
app.config["DEBUG"] = True

app.logger.addHandler(logging.StreamHandler(sys.stdout))
app.logger.setLevel(logging.ERROR)

def init():
    global d,loaded_model,service
    # load the pre-trained Keras model
    d = []
    dataFile = open('output1.txt', 'rb')
    d = pickle.load(dataFile)
    filename = 'finalized_model_rbf.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    API_KEY='AIzaSyCZspzx7MtubROWWX9NK-USz91ZeIpojoE'
    service = discovery.build('commentanalyzer', 'v1alpha1', developerKey=API_KEY)


@app.route('/', methods=['GET'])
def home():
    return '''<h1>S.A.R.C.A.S.M.A.N.I.A</h1>
<p>Can you BEEEEEEEE more sarcastic??!!.</p><p>A prototype API for sarcasmania.</p>'''

@app.errorhandler(404)
def page_not_found(e):
    return "<h1>404</h1><p>The resource could not be found.</p>", 404

@app.route('/api/sarcasmania', methods=['GET'])
def api_text():
    inputsen=""
    if 'text' in request.args:
        inputsen = (request.args['text'])
    else:
        return "Error: No text field provided. Please specify text."
    print("Input Line: ", inputsen)
    print("Please wait while the Sarcasm Data-Model loads!...")
    d = []
    dataFile = open('output1.txt', 'rb')
    d = pickle.load(dataFile)
    filename = 'finalized_model_rbf.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    API_KEY='AIzaSyCZspzx7MtubROWWX9NK-USz91ZeIpojoE'
    service = discovery.build('commentanalyzer', 'v1alpha1', developerKey=API_KEY)
    sarcasmscore = sarcasm_test().use_neural_network(inputsen)
    t= create_tfidf_training_data(d, inputsen)
    # result = loaded_model.predict(t)
    lol = loaded_model.predict_proba(t)
    humorscore = int(abs(lol[0][1]*100))

    analyze_request = {
        'comment': { 'text': inputsen },
        'requestedAttributes': {'TOXICITY': {}}
    }
    response = service.comments().analyze(body=analyze_request).execute()
    insultscore = int(abs(response['attributeScores']['TOXICITY']['summaryScore']['value']*100))
    results = {
     'Input': inputsen,
     'Sarcasm': sarcasmscore,
     'Humor': humorscore,
     'Insult': insultscore,
    }
    return jsonify(results)

def create_tfidf_training_data(docs, input):
    y = [d[0] for d in docs]
    corpus = [d[1] for d in docs]
    #corpus.append("helluss") #jo sentence hai wo yahan input dena ho ga for it to get vectorized
    vectorizer = TfidfVectorizer(min_df=1)
    X = vectorizer.fit_transform(corpus)
    t=vectorizer.transform([input])
    return t

class sarcasm_test:
    # Build the structure of the neural network exactly same as the
    # trainAndTest.py, so that the input features can be run through the neural
    #  network.
    def __init__(self):
        tf.reset_default_graph()

        number_nodes_HL1 = 100
        number_nodes_HL2 = 100
        number_nodes_HL3 = 100

        self.x = tf.placeholder('float', [None, 23])
        self.y = tf.placeholder('float')

        # Context Modelling
        with tf.name_scope("HiddenLayer1"):
            self.hidden_1_layer = {'number_of_neurons': number_nodes_HL1,
                                   'layer_weights': tf.Variable(
                                       tf.random_normal([23, number_nodes_HL1])),
                                   'layer_biases': tf.Variable(
                                       tf.random_normal([number_nodes_HL1]))}

        # User Embedding
        with tf.name_scope("HiddenLayer2"):
            self.hidden_2_layer = {'number_of_neurons': number_nodes_HL2,
                                   'layer_weights': tf.Variable(
                                       tf.random_normal(
                                           [number_nodes_HL1, number_nodes_HL2])),
                                   'layer_biases': tf.Variable(
                                       tf.random_normal([number_nodes_HL2]))}

        # Discourse Vector
        with tf.name_scope("HiddenLayer3"):
            self.hidden_3_layer = {'number_of_neurons': number_nodes_HL3,
                                   'layer_weights': tf.Variable(
                                       tf.random_normal(
                                           [number_nodes_HL2, number_nodes_HL3])),
                                   'layer_biases': tf.Variable(
                                       tf.random_normal([number_nodes_HL3]))}

        # MultiView Fusion
        with tf.name_scope("OutputLayer"):
            self.output_layer = {'number_of_neurons': None,
                                 'layer_weights': tf.Variable(
                                     tf.random_normal([number_nodes_HL3, 2])),
                                 'layer_biases': tf.Variable(tf.random_normal([2])), }
        variable_list = tf.global_variables()
        print(variable_list)
        # variable_list.remove('HiddenLayer1')
        # self.saver = tf.train.Saver(var_list=variable_list)
        self.saver = tf.train.Saver(var_list=variable_list)
        # self.saver.save(sess, 'my-model', global_step=step)

    # Nothing changes in this method as well.
    def neural_network_model(self, data):
        l1 = tf.add(tf.matmul(data, self.hidden_1_layer['layer_weights']),
                    self.hidden_1_layer['layer_biases'])
        l1 = tf.nn.relu(l1)

        l2 = tf.add(tf.matmul(l1, self.hidden_2_layer['layer_weights']),
                    self.hidden_2_layer['layer_biases'])
        l2 = tf.nn.relu(l2)

        l3 = tf.add(tf.matmul(l2, self.hidden_3_layer['layer_weights']),
                    self.hidden_3_layer['layer_biases'])
        l3 = tf.nn.relu(l3)

        output = tf.matmul(l3, self.output_layer['layer_weights']) + self.output_layer[
            'layer_biases']

        return output

    def use_neural_network(self, input_data):
        """
        In this method we restore the model created previously and obtain a
        prediction for an input sentence.
        :param input_data:
        :return:
        """
        prediction = self.neural_network_model(self.x)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            self.saver.restore(sess, 'model/sarcasm_model')
            variable_list = tf.global_variables()
            print(variable_list)
            features = CreateFeatureSet().extract_feature_of_sentence(input_data)
            pred = prediction.eval(feed_dict={self.x: [
                features]})
            result = int(2.0*(1.0/(1.0+np.exp(-pred[0][0]/100))-0.5)*100.0)
        return abs(result)



# if this is the main thread of execution first load the model and then start the server
if __name__ == "__main__":
    print(("* Loading model and Flask starting server..."
"please wait until server has fully started"))
    init()
    app.run(threaded=True)
