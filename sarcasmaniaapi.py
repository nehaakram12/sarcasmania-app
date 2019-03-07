import flask
from flask import request, jsonify
import requests
from google.cloud import storage
import google.protobuf
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

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # Explicitly use service account credentials by specifying the private key file.
    storage_client = storage.Client.from_service_account_json(
        'sarcasmaniacloudkey.json')

    # Make an authenticated API request
    buckets = list(storage_client.list_buckets())
    print(buckets)

    # storage_client = storage.Client(project="sarcasmania",credentials="OAUTH2_CREDS")
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.get_blob(source_blob_name)

    blob.download_to_filename(destination_file_name)

    print('Blob {} downloaded to {}.'.format(
        source_blob_name,
        destination_file_name))

def init():
    global d,loaded_model,service
    # load the pre-trained Keras model
    d = []
    dataFile = open('output1.txt', 'rb')
    # download_blob('staging.sarcasmania.appspot.com', 'output1.txt', 'myblob.txt')
    # dataFile = open('myblob.txt', 'rb')
    d = pickle.load(dataFile)
    filename = 'finalized_model_rbf.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    API_KEY='AIzaSyCZspzx7MtubROWWX9NK-USz91ZeIpojoE'
    service = discovery.build('commentanalyzer', 'v1alpha1', developerKey=API_KEY)


@app.route('/', methods=['GET'])
def home():
    location = requests.get('https://ipinfo.io/')
    location_data = location.json()
    city = location_data['city']
    country = location_data['country']
    weather = requests.get('http://api.openweathermap.org/data/2.5/weather?q='+city+','+country+'&appid=d583544f1369c5dfb4016d7d3ef47e0e&units=imperial')

    weather_data = weather.json()
    temperature = weather_data['main']['temp']

    response = "Current Temperature (F) in "
    response += str(city)
    response += " is: "
    response += str(temperature)

    return (response)
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
        self.saver = tf.train.Saver()
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
            self.saver.restore(sess, os.path.join(os.getcwd(),
                                                  'model/sarcasm_model'))
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
