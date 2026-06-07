import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from sklearn.model_selection import train_test_split

# 1. Configuration - Removed 'background'
DATASET_PATH = 'dataset'
CLASSES = ['siren', 'screaming', 'dog_barking', 'car_horn', 'fire_alarm']
SAMPLE_RATE = 16000

# 2. Load YAMNet from TF Hub for feature extraction
print("Loading YAMNet...")
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

def load_and_preprocess_audio(file_path):
    """Loads audio and ensures it matches YAMNet's 16kHz mono format."""
    wav, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    return wav

# 3. Extract Embeddings (Feature Extraction)
X = []  # To store 1024-dimensional embeddings
y = []  # To store integer labels (0 to 4)

print("Extracting features from audio files...")
for class_index, class_name in enumerate(CLASSES):
    class_folder = os.path.join(DATASET_PATH, class_name)
    if not os.path.exists(class_folder):
        print(f"Warning: Folder '{class_folder}' not found. Skipping.")
        continue
        
    for file_name in os.listdir(class_folder):
        if file_name.endswith('.wav'):
            file_path = os.path.join(class_folder, file_name)
            try:
                # Load audio
                wav = load_and_preprocess_audio(file_path)
                
                # Get YAMNet embeddings
                _, embeddings, _ = yamnet_model(wav)
                
                # Average the frames
                mean_embedding = tf.reduce_mean(embeddings, axis=0).numpy()
                
                X.append(mean_embedding)
                y.append(class_index)
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

X = np.array(X)
y = np.array(y)

# 4. Split into Train and Validation Sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Build the Custom Classifier Model
# The output layer automatically becomes 5 nodes because of len(CLASSES)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1024,)),             
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),                     
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(len(CLASSES), activation='softmax')  
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 6. Train the Model
print("Starting training...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=16
)

# 7. Save Your Trained Model
model.save('custom_sound_recognition_model.h5')
print("Model saved successfully as custom_sound_recognition_model.h5")