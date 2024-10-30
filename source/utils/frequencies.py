import numpy as np
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
import sys
from source import dataset_from_troll
from scipy.signal import csd, coherence
from scipy.signal import butter, filtfilt, correlate
from tqdm import tqdm

### Frequencies analysis
'''
I want to show that SCM and TROLL forces (traction force, here) are correlated even if we take separatelly different frequencies, therefore multi-fidelity usage is justified. And we don't need to learn an additional dependencies (like some unpresnet frequencies) between the lower and higher fidelities in our MF ML model? Or on the contrary and this shows some specific dependenies between the signals? 

### Cross Spectrum Analysis/Spectral coherence
https://atmos.uw.edu/~dennis/552_Notes_6c.pdf

https://en.wikipedia.org/wiki/Coherence_(signal_processing)

(Power) Spectral density - density over the frequencies present in the signal (related to auto-correlation). Cross spectral density - between signals (related to the cross-correlation).
What does this plot shows exactly...? 
Cross-correlation of signals in frequency domain. The CSD of a signal pair is the Fourier transform of the pair’s cross-correlation.
* To determine coincidence, use cross spectral density.
'''

## Cross-power spectral density
def cross_spec():
    f, Pxy = csd(troll_force, scm_force.flatten())
    plt.figure(figsize=(15, 5))
    plt.semilogy(f, np.abs(Pxy)) # set log-scale
    plt.xlabel('frequency [Hz]')
    plt.ylabel('CSD [V**2/Hz]')
    plt.show()

'''
Coherence: The coherence function estimates the extent to which a measurement of y(t) can be predicted using a measurement of x(t) with a linear function, meaning this is the time series analog of the coefficient of determination from Gaussian statistics

Coherence (x, y) = CSD(x,y)**2/(Spectral density(x)*Spectral density(y))

* To determine linearity, use coherence.
'''

'''
**Note that all of the above quantities are defined in the frequency domain based on measurements in the time domain. These values tell you about dependencies between signals in specific bandwidths.**

correlate prediction and real simulation after cutting signal's frequencies from different sides...? like, correlate pred<1Hz with real<1Hz, then 2>pred>1Hz with 2>real>1Hz, and so on...
'''

def visualization():
    f, Cxy = coherence(troll_force, scm_force.flatten())
    plt.figure(figsize=(15, 5))
    plt.semilogy(f, Cxy)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('Coherence')
    plt.show()

    # Define the bandpass filter parameters
    fs = 1000  # Sampling frequency
    corr_vec = []
    cross_corr_vec = []

    for i in tqdm(range(1, 50, 2)):
        low_cutoff = i  # Lower cutoff frequency in Hz
        high_cutoff = i+2  # Upper cutoff frequency in Hz
        filter_order = 3  # Filter order

        signal1 = troll_force 
        signal2 = scm_force.flatten()

        # Apply the bandpass filter to both signals
        b, a = butter(filter_order, [low_cutoff/(fs/2), high_cutoff/(fs/2)], btype='band')
        filtered_signal1 = filtfilt(b, a, signal1)
        filtered_signal2 = filtfilt(b, a, signal2)

        # Plot the original and filtered signals, as well as the correlation

        #plt.figure(figsize=(20,10))
        #plt.subplot(311)
        #plt.plot(signal1)
        #plt.plot(signal2)
        #plt.title('Original signals')
        #plt.subplot(312)
        #plt.plot(filtered_signal1)
        #plt.plot(filtered_signal2)
        #plt.title('Filtered signals')
        #plt.tight_layout()
        #plt.show()
        corr_vec.append(np.corrcoef(filtered_signal1, filtered_signal2)[0, 1])
        # cross-correlation is not normalized, and works as a convolution - value grows with the growing of the "shared area", but the absolute value depends on the absolute values of the initial functions
        cross_corr_vec.append(np.correlate(filtered_signal1, filtered_signal2)[0])

        plt.figure(figsize=(15,5))
        plt.scatter(range(1, 50, 2), corr_vec)
        plt.xlabel("Lower bound of a frequency bandwidth")
        plt.ylabel("Linear correlation")
        plt.xticks(np.arange(0, 50, 1))
        plt.yticks(np.arange(-1, 1, 0.1))
        plt.grid()


        plt.figure(figsize=(15,5))
        plt.scatter(range(1, 50, 2), cross_corr_vec)
        plt.xlabel("Lower bound of a frequency bandwidth")
        plt.ylabel("Cross-correlation")
        plt.xticks(np.arange(0, 50, 1))
        #plt.yticks(np.arange(0, 1, 0.1))
        plt.grid()