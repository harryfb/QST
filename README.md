# QST: Quick Stock Take
QST or Quick Stock Take is a, web based API for the recognising food objects and detecting their expiration
dates.

The current version of the model has been trained on 10 classes:

* Baked Beans
* Bread
* Butter
* Cereal
* Condiments
* Eggs
* Milk
* Pasta
* Soup
* Tinned Tomatoes

Test results indicate a mean average precision of 91.35.


## Installation
This git repository is the only file required to deploy the API. First, clone the repository locally:

```bash
git clone https://github.com/harryfb/QST.git
```

Once downloaded, navigate to the root folder of the app, *QST*, and install its dependencies:

```bash
pip install -r requirements.txt
```

Install the detectron2 library from the FAIR git repository:

```bash
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
```

Download the pytorch model file (too large to be hosted on gitHub):

```bash
wget -O ./resources/model_final.pth 'https://www.dropbox.com/s/ifghoc72yikpctj/model_final.pth?raw=1'
```

Sign up for a Google Cloud account if you do not already have one and follow their 
[guide](https://cloud.google.com/docs/authentication/getting-started) to generate a **JSON** credentials file.

Rename the JSON file to '*auth.json*' if it is not already called this and save it in the root app directory, *QST*.

Add the path to the *auth.json* file your environment variables:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="auth.json"
```

## Usage

### API

#### Requests
The API has three endpoints:
* */object-detection*
* */expiry-detection*
* */detection*

The API will only accept POST requests and returns a JSON object, containing the prediction, in response.

Endpoint | Description
---------| -------------
/object-detection | Returns food class predictions for single-item or multi-item images
/expiry-detection | Returns expiration date prediction for a single-item image
/detection | Returns both the food class prediction and expiration date prediction for a single-item image

For each, your image must be sent in the form data of the request, using '*image*' as the key and the image in byte 
format as the value:
 
    'image': <Bytes>
    
#### Response
The format JSON response object will depend on the endpoint the request is sent to. The object detector will output
three sets of predictions; the prediction from the deep learning model, the raw prediction from the textual analysis 
and a final prediction, made taking both into account. Also contained is a list of matched words and their classes and
a distance value, defined as the difference between the highest scoring and second highest scoring classes in the
textual analysis. 


#### Example I/O
This example shows the JSON object returned from the image below after requesting the */detection* endpoint.

<p align="center">
  <img src="https://github.com/harryfb/QST/blob/master/images/readme_example.jpg?raw=true"
  alt="Example of full object & expiry recognition"/>
</p>

```json
{
    "object_score": 0.6308449506759644,
    "object_class": "milk",
    "bbox": [
        19.18320083618164,
        1.2726759910583496,
        2976.383544921875,
        2325.646728515625
    ],
    "matched_words": {
        "butter": [
            "spread",
            "slightly",
            "salted",
            "lurpak",
            "butter"
        ]
    },
    "text_class": "butter",
    "text_score": 1.0,
    "text_distance": 5,
    "predicted_class": "butter",
    "predicted_score": 1.0,
    "expiry_date": "2020-07-22 00:00:00"
}
```

#### Dictionary
Textual predictions are made using a supporting JSON keyword dictionary. Adding more keywords that commonly appear on
the labels of food items within the respective classes should further improve the detectors performance.


### Python Modules
#### Import
The API has been designed as a top level component and makes use of an object detection module and an expiry date
module. These can be imported into your own .py files:

```python
from recognition.objectRec import ObjectInference
from recognition.expiryRec import ExpiryInference
```

#### Object Detection
Object recognition predictions can be generated by first running the inference on an image. This gets the raw
predictions from the deep learning model. Next, we get the final predictions, taking into account the textual content
of the image:

```python
inference = ObjectInference()

# image_bytes => (Bytes) image in bytes format
# returns     => (list) raw list of predictions from model
predictions = objectInference.run_inference(image_bytes)

# predictions => (list) raw list of predictions from model
# returns     => (list) list of predictions enhanced by text detection
predictions = objectInference.final_predictions(predictions)
```

Sometimes, only a single detection per image may be desired e.g. if attempting to detect an expiry date. In these cases,
the prediction list can be reduced by picking the one with the largest mask and (hopefully) the desired object:

```python
# predictions => (list) list of predictions
# returns     => (dict) the prediction with the largest mask area
prediction = inference.get_largest_item(predictions)
```


#### Expiry Date Detection
When running an expiry inference on an image, there is an optional argument of the objects class. Adding a class string 
will allow the detector to filter out unreasonable date predictions e.g. a date two years away for a carton of milk. 
This is most useful when running the expiration detection after object inference.

```python
expiryInference = ExpiryInference()

# image_bytes  => (Bytes) image in bytes format
# object_class => (str) class of object if known e.g. by running objectInference
# returns      => (Datetime) an expiration date prediction
prediction = expiryInference.run_inference(image_bytes, object_class)
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
