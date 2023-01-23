'''
Music analysis and synthesis via peak-finding
'''
import numpy as np
from scipy.io import wavfile
from audio_util import generate_sin
from audio_util import write_wav_16_bits
from audio_util import plot_spectrogram
from audio_util import sin_size
from matplotlib import pyplot as plt
from scipy.signal import find_peaks
from statistics import mean



def load_signal(file_name):
    # open the WAV file, confirm it is mono and read the signal
    sample_rate, original_signal = wavfile.read(file_name)
    num_channels = len(original_signal.shape)
    if num_channels != 1:
        raise Exception("Signal must be mono!")
    #if original_signal is represented in 16 bits, convert to real numbers to facilitate manipulation
    signal = original_signal.astype(float)
    #normalize it to have amplitues in the range [-1, 1]
    signal /= np.max(np.abs(signal))
    return signal, sample_rate

def peak_finding(signal, sampling_interval, sample_rate, duration, num_fft_points, spectrum_resolution):

    powerSpectrum, frequency, time, imageAxis = plt.specgram(signal, NFFT=num_fft_points, scale='dB', Fs=sample_rate, mode='magnitude', noverlap=None, vmin=None)
    num_windows = len(powerSpectrum[1]) #total number of observation windows
    audio_size = sin_size(sampling_interval, duration, initial_phase=0)
    output_signal = np.zeros((audio_size * num_windows))

    #MY PART OF THE CODE
    
    '''** What i'm actually doin is trying to verify the relevant notes in the signal.
    Now, the program can knows, for it self, how many substantial notes exists for each 
    window signal. This is a calculus based on mean.
    
    Now, i will set the duration of each substantial note. For this, i will consider 
    the powerfull of each relevant note that will be played. I mean, if in 0.15 seconds
    i'm playing one frequency and after 0.10 seconds the same note has not the same power 
    (less than a specificy and theorically based parameter), it means that this note is not more significant. 
    In this case, theres no sound in the orinal sound and the sine waves gets off. 
    This will not interfere on the dynamic of each note of each window.
    '''
    
    '''Here we find the number of significant notes and save it in a array'''
    signal_mean = np.zeros(num_windows) #A Vector that will store the signal mean of each window   
    cte=150     
    num_notes2 = np.zeros(num_windows)#A Vector that will store the number of significant notes for each window  
    for k in range(num_windows):
        temp_power_spectrum=0 
        signal_mean[k]=mean(powerSpectrum.T[k])#calculating the signal mean for each window           
        for s in range(powerSpectrum.shape[0]):
            if powerSpectrum[s][k]>=(signal_mean[k]*cte) and powerSpectrum[s][k]>= 0.01: #verifying if the current signal is relevant
                num_notes2[k]=num_notes2[k]+1
                if temp_power_spectrum==0:
                    temp_power_spectrum = powerSpectrum[s][k]
                if powerSpectrum[s][k] > temp_power_spectrum:
                    temp_power_spectrum = powerSpectrum[s][k]
                    

    ''' This part of the code was made to know the duration of each note, but it is not working yet'''
                                    
    
    '''------------------------------------------------------------------------------'''
    '''Here we find the frequency of each significant note and save it in a matrix'''
    '''
    e=0
    freq_actual=np.zeros((int(np.max(num_notes2)),num_windows))

    for q in range(num_windows):           
        for w in range(powerSpectrum.shape[0]):
            if powerSpectrum[w][q]>=(signal_mean[q]*cte): #verifying if the actual signal is relevant    
                print(freq_actual)                    
                freq_actual[e][q]=frequency[w]
                e=e+1
                print("\n"*130)
        e=0
    '''          
    '''---------------------------------------------------------------------------------------'''        
    '''
    duration=np.zeros((int(np.max(num_notes2)),num_windows))
    actual_index=0
    x=0
    temp_power=0
    for u in range(num_windows):
        for l in range(powerSpectrum.shape[0]):   
            if powerSpectrum[l][u]>=(signal_mean[u]*cte):
                if temp_power==0:
                    actual_index=l
                    temp_power=powerSpectrum[l][u]                                    
                else:
                    duration[x][u]=float((l-actual_index)*(0.25/5513))
                    actual_index=l
                    temp_power=powerSpectrum[l][u]
                    x=x+1

            if powerSpectrum[l][u] <= temp_power*0.000000005:
                duration[x][u]= float((l-actual_index)*(0.25/5513))
                x=x+1
                temp_power=0  
        x=0
    
    '''
    current_sample = 0 
    for i in range(num_windows): #iterate over each observation window
        peaks, _ = find_peaks(powerSpectrum[:,i]) #find the peak frequencies 
        sortead_peak_index = np.argsort(powerSpectrum[peaks,i]) #sort the peak frequencies array in ascending order       
        for j in range(int(num_notes2[i])): #iterate over each note to generate a sine signal
            frequency_index = peaks[sortead_peak_index[-(j+1)]] #get the index of the most powerful frequencies            
            frequency_Hz = frequency_index * spectrum_resolution #convert from index to Hz
            note = generate_sin(frequency_Hz, sampling_interval, duration, initial_phase=0)
            last_sample = current_sample + (note.shape[0])
            output_signal[current_sample:last_sample] += note
        current_sample += (note.shape[0])
    return output_signal

def main():
    #input_file_name = '../test_wav_files/music_one_note.wav'
    #input_file_name = '../test_wav_files/music_two_note.wav'
    input_file_name = 'Improving_audio_process\wav_files\sample-16bits.wav'
    signal, sample_rate = load_signal(input_file_name)
    duration = 0.25 #Duration Variable - *Working in progress* : i will make this a function that can controls the bpm of the sound.
    Ts = 1.0 / sample_rate
    window_length = 0.25 #observation window size in seconds
    num_fft_points = int(window_length / Ts)
    spectrum_resolution = sample_rate / num_fft_points
    output_signal = peak_finding(signal, Ts, sample_rate, duration, num_fft_points, spectrum_resolution)
    plot_spectrogram(output_signal, sample_rate)
    plt.show()
    #save the song in a WAV file
    #file_name = '../test_wav_files/music_one_note_after_synthesis.wav'
    #file_name = '../test_wav_files/music_two_note_after_synthesis.wav'
    file_name = 'Improving_audio_process\wav_files\Albuquerque_audio_after_synthesis.wav'
    write_wav_16_bits(file_name, sample_rate, output_signal)
    print("Wrote file", file_name)

if __name__ == "__main__":
    main()
    
    
    
    
    
    
