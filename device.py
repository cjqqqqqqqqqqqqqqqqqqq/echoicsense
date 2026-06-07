import sounddevice as sd

def list_audio_devices():
    print("\n" + "="*60)
    print("               AVAILABLE AUDIO DEVICES               ")
    print("="*60)
    
    # Fetch the complete device array from the system host API
    devices = sd.query_devices()
    default_input = sd.query_hostapis()[0]['default_input_device']
    default_output = sd.query_hostapis()[0]['default_output_device']
    
    print(f"{'Index':<7} | {'Device Name':<45} | {'Inputs':<8} | {'Outputs':<8}")
    print("-" * 75)
    
    for index, device in enumerate(devices):
        # Mark default devices with clean indicators
        name = device['name']
        if index == default_input and index == default_output:
            name += " (Default In/Out)"
        elif index == default_input:
            name += " (Default Input 🎙️)"
        elif index == default_output:
            name += " (Default Output 🔊)"
            
        max_inputs = device['max_input_channels']
        max_outputs = device['max_output_channels']
        
        print(f"{index:<7} | {name:<45} | {max_inputs:<8} | {max_outputs:<8}")
        
    print("="*75 + "\n")

if __name__ == "__main__":
    list_audio_devices()