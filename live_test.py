import time
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import sounddevice as sd
from scipy.signal import butter, lfilter

# ==========================================
# 1. HARDWARE SELECTION FUNCTION
# ==========================================
def select_microphone():
    print("\n" + "="*50)
    print("             AVAILABLE MICROPHONES             ")
    print("==============================================")
    devices = sd.query_devices()
    default_input = sd.query_hostapis()[0]['default_input_device']
    valid_mic_indexes = []
    
    print(f"{'Index':<7} | {'Microphone Name':<45}")
    print("-" * 55)
    for index, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            valid_mic_indexes.append(index)
            name = device['name']
            if index == default_input:
                name += " ⭐ (System Default)"
            print(f"{index:<7} | {name:<45}")
    print("="*50)
    
    user_input = input(f"Select Microphone Index [Default is {default_input}]: ").strip()
    if user_input == "":
        return default_input
    try:
        chosen_index = int(user_input)
        return chosen_index if chosen_index in valid_mic_indexes else default_input
    except ValueError:
        return default_input

CHOSEN_MIC_INDEX = select_microphone()

# ==========================================
# 2. CONFIGURATION, THRESHOLDS & MODELS
# ==========================================
CLASSES = ['siren', 'screaming', 'dog_barking', 'car_horn', 'fire_alarm']
SAMPLE_RATE = 16000
DURATION = 1.0

# Your optimized operational thresholds
CONFIDENCE_THRESHOLD = 0.60
ENERGY_THRESHOLD = 0.005

print("Loading sound classification models...")
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
custom_model = tf.keras.models.load_model('custom_sound_recognition_model.h5')
print("Noise-filtered terminal testing loop active. Listening...")

# ==========================================
# 3. NOISE REDUCTION FILTER (DSP LAYER)
# ==========================================
def butter_highpass(cutoff, fs, order=5):
    """Calculates the mathematical coefficients for a Butterworth filter."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, cutoff=300, fs=16000, order=5):
    """Applies the high-pass filter to block low-end background rumble."""
    b, a = butter_highpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y.astype('float32')

# ==========================================
# 4. LIVE RECORDING LOOP
# ==========================================
while True:
    try:
        # Record raw audio snippet via selected device index
        recording = sd.rec(
            int(DURATION * SAMPLE_RATE), 
            samplerate=SAMPLE_RATE, 
            channels=1, 
            dtype='float32',
            device=CHOSEN_MIC_INDEX
        )
        sd.wait()
        
        raw_wav = np.squeeze(recording)
        
        # --- APPLY DIGITAL NOISE REDUCTION ---
        # Slices out baseline continuous noise signatures below 300Hz
        filtered_wav = highpass_filter(raw_wav, cutoff=300, fs=SAMPLE_RATE)
        
        # Calculate root-mean-square loudness on the cleaned wave array
        rms = np.sqrt(np.mean(np.square(filtered_wav)))
        
        # Verify signal activity against cleaned sound metric
        if rms < ENERGY_THRESHOLD:
            continue
            
        # Extract features using filtered array data blocks
        _, embeddings, _ = yamnet_model(filtered_wav)
        if embeddings.shape[0] == 0:
            continue
            
        mean_embedding = tf.reduce_mean(embeddings, axis=0).numpy()
        mean_embedding = np.expand_dims(mean_embedding, axis=0)
        
        predictions = custom_model.predict(mean_embedding, verbose=0)[0]
        highest_score_index = np.argmax(predictions)
        confidence = predictions[highest_score_index]
        predicted_class = CLASSES[highest_score_index]
        
        if confidence > CONFIDENCE_THRESHOLD:
            print(f"[{time.strftime('%H:%M:%S')}] DETECTED: {predicted_class.upper()} ({confidence*100:.1f}%)")
            
    except KeyboardInterrupt:
        print("\nStopping live test.")
        break
    except Exception as e:
        print(f"Error in Live Test Loop: {e}")
        time.sleep(1)