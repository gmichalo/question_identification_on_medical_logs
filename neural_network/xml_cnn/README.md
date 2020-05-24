# XML_CNN

This is an implementation for Deep Learning for Extreme Multi-label Text Classification [XML_CNN](http://nyc.lti.cs.cmu.edu/yiming/Publications/jliu-sigir17.pdf)

## Running code

### Hyperpameter tuning
In order to search for the best value for the hyperparameter of the model (50 trials) :
- First, create a json file like the file "search_spaces/cnn.json" for all the hyperparameters that you want to tune with name xml_cnn.json

- Then the following code in the previous directory
```
python3 param_json.py --model_name "XML_CNN"  -fn "XML_CNN" - -jf "search_spaces/xml_cnn.json" -st 50
```
The best hyperparameters will  be saved in *dataset_output/hyperpameters/xml_cnn.json*
### Running model
In order to run this model run the following code in the previous directory
```
python3 main_iterations.py --model_name "XML_CNN"  -fn "XML_CNN"
```
