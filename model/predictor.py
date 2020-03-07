import os
import io
from uuid import uuid1
import flask
from PIL import Image
import autogluon as ag
from autogluon.task.image_classification import Classifier
import json

model_path = os.environ['MODEL_PATH']


# A singleton for holding the model. This simply loads the model and holds it.
# It has a predict function that does a prediction based on the model and the input data.
class AutoGluonClassifierService(object):
    """
    Singleton for holding the AutoGluon Tabular task model.
    It has a predict function that does inference based on the model and input data
    """
    model = None

    @classmethod
    def load_model(cls) -> Classifier:
        """Load AutoGluon Tabular task model for this instance, loading it if it's not already loaded."""
        if cls.model is None:
            cls.model = Classifier.load(model_path)
        return cls.model

    @classmethod
    def predict(cls, image_path: str) -> tuple:
        """For the input, do the predictions and return them.
        Args:
            image_path (a str): Path to image"""

        print("Classify input image: ")
        return cls.model.predict(image_path)


# The flask app for serving predictions
app = flask.Flask(__name__)


@app.route('/ping', methods=['GET'])
def ping():
    """Determine if the container is working and healthy. In this sample container, we declare
    it healthy if we can load the model successfully."""
    health = AutoGluonClassifierService.load_model() is not None  # You can insert a health check here

    status = 200 if health else 404
    return flask.Response(response='\n', status=status, mimetype='application/json')


@app.route('/invocations', methods=['POST'])
def transformation():
    """Do an inference on a single batch of data. In this sample server, we take data as CSV, convert
    it to a pandas data frame for internal use and then convert the predictions back to CSV (which really
    just means one prediction per line, since there's a single column.
    """
    AutoGluonClassifierService.load_model()
    data = None
    print(f'Request Content Type: {flask.request.content_type}')
    # Convert from CSV to pandas
    if flask.request.content_type == 'application/x-image':
        data = flask.request.data.decode('utf-8')
        tmp_image_path = f'/tmp/{uuid1().hex}.jpg'
        image_bytes = io.BytesIO(data)
        image = Image.open(image_bytes)
        image.save(tmp_image_path)
    else:
        return flask.Response(
            response='This predictor only supports JSON or CSV data.  data is preferred.',
            status=415, mimetype='text/plain'
        )

    print('Classifying image with {}')
    # Do the prediction
    class_index, class_probability = AutoGluonClassifierService.predict(tmp_image_path)
    prediction = {
        'ClassIndex': class_index,
        'PredictionProba': class_probability
    }

    return flask.Response(response=json.dumps(prediction), status=200, mimetype='application/json')
