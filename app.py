import time
import threading
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import sounddevice as sd
from scipy.signal import butter, lfilter
from flask import Flask, render_template_string, jsonify
from collections import deque

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
# 2. CONFIGURATION & GLOBAL STATE
# ==========================================
CLASSES = ['siren', 'screaming', 'dog_barking', 'car_horn', 'fire_alarm']
SAMPLE_RATE = 16000

DURATION = 1.5
LOOP_BREAK_DURATION = 0.5

CONFIDENCE_THRESHOLD = 0.60
ENERGY_THRESHOLD = 0.005

# Rolling window history buffer (Tracks last 3 iterations of predictions)
prediction_history = deque(maxlen=3)

latest_alert = {"class": None, "timestamp": None}

app = Flask(__name__)

# ==========================================
# 3. NOISE REDUCTION & DSP EXTRACTION LAYER
# ==========================================
def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def highpass_filter(data, cutoff=300, fs=16000, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    return lfilter(b, a, data).astype('float32')

def extract_dsp_features(signal):
    # 1. Zero Crossing Rate (ZCR)
    zcr = np.mean(np.abs(np.diff(np.sign(signal))) > 0)
    
    # 2. Spectral Flatness
    fft_vals = np.abs(np.fft.rfft(signal)) + 1e-10
    geometric_mean = np.exp(np.mean(np.log(fft_vals)))
    arithmetic_mean = np.mean(fft_vals)
    spectral_flatness = geometric_mean / arithmetic_mean
    
    return zcr, spectral_flatness

# ==========================================
# 4. AUDIO CLASSIFICATION WORKER THREAD
# ==========================================
def audio_processing_worker():
    global latest_alert
    print("[Thread 1] Loading sound classification models...")
    yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')
    custom_model = tf.keras.models.load_model('custom_sound_recognition_model.h5')
    print(f"[Thread 1] Active. Settings: {DURATION}s record, {LOOP_BREAK_DURATION}s break.")

    while True:
        try:
            recording = sd.rec(
                int(DURATION * SAMPLE_RATE), 
                samplerate=SAMPLE_RATE, 
                channels=1, 
                dtype='float32',
                device=CHOSEN_MIC_INDEX
            )
            sd.wait()
            
            raw_wav = np.squeeze(recording)
            filtered_wav = highpass_filter(raw_wav, cutoff=300, fs=SAMPLE_RATE)
            rms = np.sqrt(np.mean(np.square(filtered_wav)))
            timestamp = time.strftime('%H:%M:%S')
            
            if rms < ENERGY_THRESHOLD:
                prediction_history.append(None)
                time.sleep(LOOP_BREAK_DURATION)
                continue
                
            zcr, flatness = extract_dsp_features(filtered_wav)
            
            _, embeddings, _ = yamnet_model(filtered_wav)
            if embeddings.shape[0] == 0:
                time.sleep(LOOP_BREAK_DURATION)
                continue
                
            mean_embedding = tf.reduce_mean(embeddings, axis=0).numpy()
            mean_embedding = np.expand_dims(mean_embedding, axis=0)
            
            predictions = custom_model.predict(mean_embedding, verbose=0)[0]
            
            screaming_idx = CLASSES.index('screaming')
            barking_idx = CLASSES.index('dog_barking')
            
            if predictions[screaming_idx] > 0.3 or predictions[barking_idx] > 0.3:
                if zcr > 0.15:
                    predictions[screaming_idx] += 0.20
                    predictions[barking_idx] -= 0.15
                elif flatness > 0.40:
                    predictions[barking_idx] += 0.20
                    predictions[screaming_idx] -= 0.15

            highest_score_index = np.argmax(predictions)
            confidence = predictions[highest_score_index]
            
            if confidence > CONFIDENCE_THRESHOLD:
                detected_class = CLASSES[highest_score_index]
                prediction_history.append(detected_class)
            else:
                prediction_history.append(None)
                
            if len(prediction_history) == prediction_history.maxlen:
                history_list = list(prediction_history)
                
                if history_list.count('screaming') >= 2:
                    latest_alert = {"class": "SCREAMING", "timestamp": timestamp}
                elif history_list[-1] == 'dog_barking':
                    latest_alert = {"class": "DOG BARKING", "timestamp": timestamp}
                elif history_list[-1] is not None and history_list[-1] not in ['screaming', 'dog_barking']:
                    latest_alert = {"class": history_list[-1].upper(), "timestamp": timestamp}
            
            time.sleep(LOOP_BREAK_DURATION)
                    
        except Exception as e:
            print(f"Error in Classification Thread: {e}")
            time.sleep(1)

# ==========================================
# 5. WEB DASHBOARD CONFIGURATION (LEFT HAND SIDE ONLY)
# ==========================================
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>EchoicSense Dashboard (Audio Only)</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; 
            background-color: #000000; color: #ffffff; 
            margin: 0; padding: 0; display: flex;
            width: 100vw; height: 100vh; overflow: hidden;
        }
        .column {
            flex: 1; display: flex; justify-content: center; align-items: center;
            text-align: center; font-size: 3.5rem; font-weight: bold; padding: 40px; word-break: break-word;
        }
        #left-side { border-right: 1px solid #111111; }
        #right-side { background-color: #000000; }
    </style>
    <script>
        setInterval(async () => {
            try {
                let response = await fetch('/api/data');
                let data = await response.json();
                document.getElementById('left-side').innerText = data.alert.class ? `"${data.alert.class}"` : '';
            } catch (err) {
                console.error(err);
            }
        }, 400);
    </script>
</head>
<body>
    <div id="left-side" class="column"></div>
    <div id="right-side" class="column"></div>
</body>
</html>
"""

@app.route('/')
def home(): 
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/data')
def get_data():
    global latest_alert
    current_time_struct = time.localtime()
    
    if latest_alert["timestamp"]:
        a_struct = time.strptime(latest_alert["timestamp"], '%H:%M:%S')
        elapsed = (current_time_struct.tm_min * 60 + current_time_struct.tm_sec) - (a_struct.tm_min * 60 + a_struct.tm_sec)
        if elapsed > 5 or elapsed < 0: 
            latest_alert = {"class": None, "timestamp": None}
            
    return jsonify({"alert": latest_alert})

if __name__ == '__main__':
    threading.Thread(target=audio_processing_worker, daemon=True).start()
    app.run(debug=False, port=5000)