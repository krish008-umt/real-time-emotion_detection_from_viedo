import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
import warnings
warnings.filterwarnings('ignore')

class FaceExpressionModel:
    def __init__(self):
        self.model = None
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.detector = MTCNN()
        self.load_model()  
    def load_model(self):
        try:
           
            file_path = r"C:\Users\Dell\Downloads\face_exp.h5"
           
            import os
            if not os.path.exists(file_path):
                print(f" Error: File not found: {file_path}")
                return
            
            print(f" File found. Size: {os.path.getsize(file_path)} bytes")
            
            
            self.model = load_model(file_path)
            print(" Model loaded successfully!")
            
           
            print("\nModel Summary:")
            print(f"Input shape: {self.model.input_shape}")
            print(f"Output shape: {self.model.output_shape}")
            
        except Exception as e:
            print(f" Error loading model: {e}")
           
            self.build_model_from_scratch()
    
    def build_model_from_scratch(self):
      
        inp = tf.keras.Input(shape=(150, 150, 3))

        x = layers.Conv2D(32, (3,3), padding="same")(inp)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(2,2)(x)
        x = layers.Dropout(0.2)(x)  

        x = layers.Conv2D(64, (3,3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(2,2)(x)
        x = layers.Dropout(0.3)(x)  

        x = layers.Conv2D(128, (3,3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(2,2)(x)
        x = layers.Dropout(0.4)(x)  

        x = layers.Conv2D(256, (3,3), padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(2,2)(x)
        x = layers.Dropout(0.4)(x)

        x = layers.Flatten()(x)

        x = layers.Dense(256)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.Dropout(0.5)(x)

        out = layers.Dense(7, activation="softmax")(x)

        self.model = tf.keras.Model(inputs=inp, outputs=out)
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        print(" Model built from scratch")
    
    def predict(self, img):
        if self.model is None:
            print("Model not loaded!")
            return img
       
        print(f"\nInput image shape: {img.shape}")
        
        
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
       
        faces = self.detector.detect_faces(rgb_img)
        
        print(f"Faces detected: {len(faces)}")
        
        if len(faces) == 0:
            print("No faces detected")
            return img
        
        for i, face_data in enumerate(faces):
           
            x, y, w, h = face_data['box']
            
           
            x, y = max(0, x), max(0, y)
            
            print(f"\nFace {i+1}:")
            print(f"  Bounding box: x={x}, y={y}, w={w}, h={h}")
            
            face_roi = img[y:y+h, x:x+w]
            
            if face_roi.size == 0:
                print(f"  Warning: Empty face region")
                continue
            
            print(f"  Face ROI shape: {face_roi.shape}")
          
            target_size = self.model.input_shape[1:3]  
            print(f"  Model expects input shape: {target_size}")
            
           
            face_resized = cv2.resize(face_roi, target_size)
            print(f"  Resized to: {face_resized.shape}")
            
           
            face_normalized = face_resized.astype('float32') / 255.0
            print(f"  Normalized range: {face_normalized.min():.2f} to {face_normalized.max():.2f}")
            
           
            face_batch = np.expand_dims(face_normalized, axis=0)
            print(f"  Batch shape: {face_batch.shape}")
            
            try:
               
                predictions = self.model.predict(face_batch, verbose=0)
                predicted_class = np.argmax(predictions[0])
                predicted_label = self.emotion_labels[predicted_class]
                confidence = predictions[0][predicted_class]
                
                print(f"  Prediction: {predicted_label} (confidence: {confidence:.2f})")
                
               
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                
                text = f"{predicted_label} ({confidence:.2f})"
                cv2.putText(img, text, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (36, 255, 12), 2)
                
            except Exception as e:
                print(f"  Prediction error: {e}")
        
        return img

if __name__ == "__main__":
  
   
    
   
    fem = FaceExpressionModel()
    
    if fem.model is None:
        print(" Could not initialize model. Exiting...")
        exit()
    
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print(" Error: Could not open camera.")
        exit()
    
    print("\n Camera opened successfully!")
    print("Press 'q' to quit\n")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        frame_count += 1
        
        if not ret:
            print(" Failed to grab frame")
            break
        
        
        frame = cv2.flip(frame, 1)
        

        if frame_count % 10 == 0:
            print(f"\nFrame {frame_count}:")
            print(f"  Shape: {frame.shape}")
            print(f"  Type: {frame.dtype}")
            print(f"  Range: {frame.min()} to {frame.max()}")
        
       
        processed_frame = fem.predict(frame)
        
        
        cv2.imshow("Face Expression Recognition", processed_frame)
        
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
   
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Program terminated.")