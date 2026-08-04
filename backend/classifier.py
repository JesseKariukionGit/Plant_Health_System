import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Dense
import json
import os

# Custom Dense layer that removes quantization_config
class CustomDense(Dense):
    def __init__(self, *args, **kwargs):
        kwargs.pop('quantization_config', None)
        super().__init__(*args, **kwargs)

class PlantDiseaseClassifier:
    def __init__(self, model_path="../models/efficientnetb0_plantvillage.h5", 
                 class_names_path="../models/class_names.json"):
        
        self.model_path = model_path
        self.class_names_path = class_names_path
        
        if os.path.exists(model_path):
            # Load with custom_objects to use CustomDense
            self.model = load_model(
                model_path,
                compile=False,
                custom_objects={'Dense': CustomDense}
            )
            print(f"✅ Model loaded from {model_path}")
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")
        
        if os.path.exists(class_names_path):
            with open(class_names_path, 'r') as f:
                self.class_names = json.load(f)
            print(f"✅ Loaded {len(self.class_names)} disease classes")
        else:
            raise FileNotFoundError(f"Class names not found at {class_names_path}")
        
        self.image_size = 224
    
    def preprocess_image(self, image_path):
        img = load_img(image_path, target_size=(self.image_size, self.image_size))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    
    def predict(self, image_path):
        processed_img = self.preprocess_image(image_path)
        predictions = self.model.predict(processed_img, verbose=0)
        
        confidence = float(np.max(predictions[0]))
        predicted_class_idx = int(np.argmax(predictions[0]))
        disease_name = self.class_names[predicted_class_idx]
        
        all_probs = {self.class_names[i]: float(predictions[0][i]) 
                     for i in range(len(self.class_names))}
        
        return disease_name, confidence, all_probs