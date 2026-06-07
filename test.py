import sys
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa

# 1. Configuration
CLASSES = ['siren', 'screaming', 'dog_barking', 'car_horn', 'fire_alarm']
SAMPLE_RATE = 16000
CONFIDENCE_THRESHOLD = 0.80  # 80% confidence needed to trigger an alert

# 2. Check if a file path was provided
if len(sys.argv) < 2:
    print("Usage: python test.py <path_to_audio_file.wav>")
    sys.exit()

TEST_FILE = sys.argv[1]

print("Loading models...")
# Load YAMNet for feature extraction
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
# Load your freshly trained custom classifier
custom_model = tf.keras.models.load_model('custom_sound_recognition_model.h5')

try:
    # 3. Load and preprocess the test audio
    print(f"Analyzing audio: {TEST_FILE}")
    wav, sr = librosa.load(TEST_FILE, sr=SAMPLE_RATE, mono=True)
    
    # 4. Extract embeddings using YAMNet
    _, embeddings, _ = yamnet_model(wav)
    mean_embedding = tf.reduce_mean(embeddings, axis=0).numpy()
    
    # Reshape because Keras expects a batch (1, 1024)
    mean_embedding = np.expand_dims(mean_embedding, axis=0)
    
    # 5. Make the prediction
    predictions = custom_model.predict(mean_embedding, verbose=0)[0]
    
    # 6. Apply Threshold Decision Logic
    highest_score_index = np.argmax(predictions)
    confidence = predictions[highest_score_index]
    predicted_class = CLASSES[highest_score_index]
    
    print("\n--- RESULTS ---")
    if confidence > CONFIDENCE_THRESHOLD:
        print(f"ALERT DETECTED: {predicted_class.upper()}")
        print(f"Confidence: {confidence * 100:.2f}%")
    else:
        print("ENVIRONMENT: Background Noise / Unrecognized Sound")
        print(f"Highest guess was '{predicted_class}' but only at {confidence * 100:.2f}% (Below Threshold)")
        
except Exception as e:
    print(f"Error analyzing file: {e}")