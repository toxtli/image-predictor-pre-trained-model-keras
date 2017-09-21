from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from keras.preprocessing.image import load_img, img_to_array
import sys, numpy as np

model = ResNet50(weights='imagenet')
x = preprocess_input(np.expand_dims(img_to_array(load_img(sys.argv[1], target_size=(224, 224))), axis=0))
preds = model.predict(x)
print('Predicted:', decode_predictions(preds, top=1)[0])