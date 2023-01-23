'''
Music analysis and synthesis via peak-finding
'''
import numpy as np
from scipy.io import wavfile
from audio_util import generate_sin
from audio_util import write_wav_16_bits
from audio_util import plot_spectrogram
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

def load_signal(file_name):
    # open the WAV file, confirm it is mono and read the signal
    sample_rate, original_signal = wavfile.read(file_name)
    #print(sample_rate)
    #print(len(original_signal.shape))
    #exit(-1)
    num_channels = len(original_signal.shape)
    #print(num_channels)
    #exit(-1)
    if num_channels != 1:
        raise Exception("Signal must be mono!")

    #if original_signal is represented in 16 bits, convert to real numbers to facilitate manipulation
    signal = original_signal.astype(float)
    #normalize it to have amplitues in the range [-1, 1]
    signal /= np.max(np.abs(signal))
    #print(len(signal))
    #exit(-1)
    return signal, sample_rate

def peak_finding(signal, sampling_interval, sample_rate, window_length, num_notes, num_fft_points, spectrum_resolution):

    powerSpectrum, frequency, time, imageAxis = plt.specgram(signal, NFFT=num_fft_points, scale='dB', Fs=sample_rate, mode='magnitude', noverlap=None, vmin=None)
    #print(len(powerSpectrum[1]));
    #print(len(frequency))
    #exit(-1)
    num_windows = len(powerSpectrum[1]) #total number of observation windows
    output_signal = np.zeros((num_fft_points * num_windows))
    media = np.zeros(num_windows)
    #print(len(output_signal))
    #exit(-1)
    current_sample = 0
    for i in range(num_windows): #iterate over each observation window
        peaks, _ = find_peaks(powerSpectrum[:,i]) #find the peak frequencies 
        print(sum(peaks)/len(peaks))
        media=sum(peaks)/len(peaks)
        for k in range(len(peaks)):
            if peaks[k] >= 1.7*media:
                num_notes2+=num_notes2
        #exit(-1)
        sortead_peak_index = np.argsort(powerSpectrum[peaks,i]) #sort the peak frequencies array in ascending order
        #print(sortead_peak_index)
        #exit(-1)  
        for j in range(num_notes - 1): #iterate over each note to generate a sine signal
            frequency_index = peaks[sortead_peak_index[-(j+1)]] #get the index of the most powerful frequencies
            print(frequency_index)
            
            frequency_index = int(frequency_index / spectrum_resolution) #convert the frequency index to an integer
            #print(frequency_index)
            #exit(-1)
            frequency_Hz = frequency_index * spectrum_resolution #convert from index to Hz
            #print(frequency_Hz)
            
            note = generate_sin(frequency_Hz, sampling_interval, window_length, initial_phase = 0) #generate the sine signal
            last_sample = current_sample + num_fft_points
            output_signal[current_sample:last_sample] += note #store all the notes in a single observation window
        current_sample += num_fft_points
       

    return output_signal

def main():
    #input_file_name = '../test_wav_files/music_one_note.wav'
    #input_file_name = '../test_wav_files/music_two_note.wav'
    input_file_name = '/home/albu/Documents/GITREP/dsp-audio-main/test_wav_files/sample-16bits.wav'

    signal, sample_rate = load_signal(input_file_name)
    
    Ts = 1.0 / sample_rate
    window_length = 0.25 #observation window size in seconds
    num_notes = 5 #number of notes to be found in each observation window
    num_fft_points = int(window_length / Ts)
    #print(num_fft_points)
    #exit(-1)
    spectrum_resolution = sample_rate / num_fft_points

    output_signal = peak_finding(signal, Ts, sample_rate, window_length, num_notes, num_fft_points, spectrum_resolution)

    #plot spectrogram
    plot_spectrogram(output_signal, sample_rate)
    plt.show()

    #save the song in a WAV file
    #file_name = '../test_wav_files/music_one_note_after_synthesis.wav'
    #file_name = '../test_wav_files/music_two_note_after_synthesis.wav'
    file_name = '/home/albu/Documents/GITREP/dsp-audio-main/test_wav_files/sample-16bits.wav'
    write_wav_16_bits(file_name, sample_rate, output_signal)
    print("Wrote file", file_name)

if __name__ == "__main__":
    main()
    
    
    
    
    
    
#*So, what about now?
#There's a lot things that we can work in this code. Actually, when we talk about sound design,
#or audio sythesis, what we propose is a harmonic audio that reproduces all the frequencies in 
#the sample. Obviously, this is impossible 'cause in a wave file we have a lot of noises
#that we want to avoid to not distort the output. 
#So, for this vision point, we can improve our fundamentals notes of each window audio.
#To do this, (1) we can compose the sine, that are being generated, with phasereds sines.
#For the second part (2), we can takeout the variables that are "controling" the program. 
#
