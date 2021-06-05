import numpy
import librosa, librosa.display as ldisp
import matplotlib.pyplot as plot

audiodata = "/content/drive/MyDrive/music/blues/blues.00000.wav"

#  load audio data with Librosa
audiosignal, samplerate = librosa.load(audiodata, sr=22050)

FIG_SIZE = (21,14)

#  WAVEFORM
#  display waveform
plot.figure(figsize=FIG_SIZE)
ldisp.waveplot(audiosignal, samplerate, alpha=0.4)
plot.xlabel("Time (s)")
plot.ylabel("Amplitude")
plot.title("Waveform")


#  FFT -> power spectrum
# perform Fourier transform
fft = numpy.fft.fft(audiosignal)

#  calculate abs values on complex numbers to get magnitude
spectrum = numpy.abs(fft)

#  create frequency variable
f = numpy.linspace(0, samplerate, len(spectrum))


#  take half of the spectrum and frequency
left_spectrum = spectrum[:int(len(spectrum)/2)]
left_f = f[:int(len(spectrum)/2)]

#  plot spectrum
plot.figure(figsize=FIG_SIZE)
plot.plot(left_f, left_spectrum, alpha=0.4)
plot.xlabel("Frequency")
plot.ylabel("Magnitude")
plot.title("Power spectrum")


#  STFT -> spectrogram
hop_length = 512 # in num. of samples
n_fft = 2048 # window in num. of samples

#  calculate duration hop length and window in seconds
hop_length_duration = float(hop_length)/samplerate
n_fft_duration = float(n_fft)/samplerate

print("STFT hop length duration is: {}s".format(hop_length_duration))
print("STFT window duration is: {}s".format(n_fft_duration))

#  perform stft
stft = librosa.stft(audiosignal, n_fft=n_fft, hop_length=hop_length)

#  calculate abs values on complex numbers to get magnitude
spectrogram = numpy.abs(stft)

#  display spectrogram
plot.figure(figsize=FIG_SIZE)
ldisp.specshow(spectrogram, sr=samplerate, hop_length=hop_length)
plot.xlabel("Time")
plot.ylabel("Frequency")
plot.colorbar()
plot.title("Spectrogram")

#  apply logarithm to cast amplitude to Decibels
log_spectrogram = librosa.amplitude_to_db(spectrogram)

plot.figure(figsize=FIG_SIZE)
ldisp.specshow(log_spectrogram, sr=samplerate, hop_length=hop_length)
plot.xlabel("Time")
plot.ylabel("Frequency")
plot.colorbar(format="%+2.0f dB")
plot.title("Spectrogram (dB)")


#  MFCCs
#  extract 13 MFCCs
MFCCs = librosa.feature.mfcc(audiosignal, samplerate, n_fft=n_fft, hop_length=hop_length, n_mfcc=13)

#  display MFCCs
plot.figure(figsize=FIG_SIZE)
ldisp.specshow(MFCCs, sr=samplerate, hop_length=hop_length)
plot.xlabel("Time")
plot.ylabel("MFCC coefficients")
plot.colorbar()
plot.title("MFCCs")

#  show plots
plot.show()