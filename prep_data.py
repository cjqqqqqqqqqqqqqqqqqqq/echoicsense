import os
import librosa
import soundfile as sf

# 1. Configuration
DATASET_PATH = 'dataset'
CLASSES = ['siren', 'screaming', 'dog_barking', 'car_horn', 'fire_alarm']
MAX_LENGTH = 10.0  # seconds

print("--- DIAGNOSTIC MODE START ---")
print(f"Looking for dataset folder at absolute path: {os.path.abspath(DATASET_PATH)}")

if not os.path.exists(DATASET_PATH):
    print(f"\nCRITICAL ERROR: The folder '{DATASET_PATH}' does not exist in this directory.")
    print("Please make sure your dataset folder is in the same place as this script.")
    exit()

for class_name in CLASSES:
    class_folder = os.path.join(DATASET_PATH, class_name)
    
    if not os.path.exists(class_folder):
        print(f"\n[WARNING] Folder missing: Cannot find '{class_folder}'. Skipping.")
        continue
        
    print(f"\nScanning folder: {class_name}...")
    files_found = False
        
    for file_name in os.listdir(class_folder):
        # Made case-insensitive to catch .WAV and .wav
        if file_name.lower().endswith('.wav'):
            files_found = True
            file_path = os.path.join(class_folder, file_name)
            
            try:
                y, sr = librosa.load(file_path, sr=None)
                duration = librosa.get_duration(y=y, sr=sr)
                
                if duration > MAX_LENGTH:
                    print(f"  -> SPLITTING: {file_name} ({duration:.1f} seconds)")
                    
                    midpoint = len(y) // 2
                    part1 = y[:midpoint]
                    part2 = y[midpoint:]
                    
                    base_name = os.path.splitext(file_name)[0]
                    part1_path = os.path.join(class_folder, f"{base_name}_part1.wav")
                    part2_path = os.path.join(class_folder, f"{base_name}_part2.wav")
                    
                    sf.write(part1_path, part1, sr)
                    sf.write(part2_path, part2, sr)
                    
                    os.remove(file_path)
                    print(f"     Success: Created part1 and part2, deleted original.")
                else:
                    print(f"  -> OK: {file_name} ({duration:.1f} seconds) - No split needed.")
                    
            except Exception as e:
                print(f"  -> ERROR processing {file_name}: {e}")
                
    if not files_found:
         print(f"  -> No .wav files found in {class_folder}.")

print("\n--- DIAGNOSTIC MODE COMPLETE ---")