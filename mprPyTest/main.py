'''
Created on Dec 28, 2016

@author: mrichwine
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import * # Grab MATLAB plotting functions
#from control.matlab import *    # MATLAB-like functions
from numpy import *

# Get Sound Device https://github.com/spatialaudio/python-sounddevice/
import sounddevice as sd

def make_wave():
    print ("Setting Up Waveforms")
    fs  = 44100
    dur = 30
    t   = np.arange(0,dur,1/fs)
    len = np.size(t,0)
        
    amplBase = np.linspace(1,1,len) #pu
    freqBase = np.linspace(60,60,len) #Hz
    
    # model contingency deviation
    k1 = 6
    t1 = 25
    k2 = 0.03
    t2 = 5
    t_blank = 5 #sec of steady-state
    
    t_dev = t - t_blank
    ss = np.zeros(t_blank*fs)
    dy = np.ones((dur-t_blank)*fs)
    mask = np.concatenate((ss,dy),axis=0)
    freqDev = mask*k1*(np.exp(-1*t_dev/t1)-1)*(np.exp(-1*t_dev/t2)+k2)
    
    # generate magnitude and phase noise
    CMnoiseGain    = 0.0#1 #pu
    DMnoiseGain    = 0.0#05 #pu
    CMnoise        = CMnoiseGain*np.random.uniform(0,1,len)
    DMnoise        = DMnoiseGain*np.random.uniform(0,1,len)
    
    CPnoiseGain    = 0.0#1 #rads
    DPnoiseGain    = 0.0#05 #rads
    CPnoise        = CPnoiseGain*np.random.uniform(0,1,len)
    DPnoise        = DPnoiseGain*np.random.uniform(0,1,len)
    
    # Generate waveforms    
    VdAmpl  = amplBase + CMnoise + DMnoise
    VqAmpl  = amplBase + CMnoise - DMnoise
    VdPhase = 2*np.pi*(freqBase + freqDev)*t + CPnoise + DPnoise
    VqPhase = 2*np.pi*(freqBase + freqDev)*t + CPnoise - DPnoise
    
    Vd = VdAmpl*np.cos(VdPhase)
    Vq = VqAmpl*np.sin(VqPhase)
    
    Va = Vd
    Vb = -0.5*Vd + (np.sqrt(3)/2)*Vq
    Vc = -0.5*Vd - (np.sqrt(3)/2)*Vq

    return Va
    
    # Plot
    plt.close('all')
    fig1 = plt.figure(1,figsize=(3*1.6,3*1.6))
    fig1.clf()
    plt.plot(t,Va)
    plt.plot(t,Vb,'r')
    plt.plot(t,Vc,'g')
    plt.grid()
    plt.xlim([0.0,0.05])
    
    """
    fig2 = plt.figure(2,figsize=(3*1.6,3*1.6))
    fig2.clf()
    plt.plot(t,noiseAmpl)
    plt.plot(t,noisePhase,'k')
    plt.grid()
    """
    fig3 = plt.figure(3,figsize=(3*1.6,3*1.6))
    fig3.clf()
    plt.plot(t,freqDev)
    plt.grid()
    #plt.show()


    
    print("waveforms ready")

def decode_wave():
    print ("Start Decoding Waveforms")
    """
    # Calculate RMS quantities
    Vd = (2*Va - Vb - Vc)/3
    Vq = (Vb - Vc)/np.sqrt(3)
    Vm = np.sqrt(Vd**2 + Vq**2)/(1000*np.sqrt(2/3)) #kV
    Vab = Va - Vb
    Vbc = Vb - Vc
    Vca = Vc - Va
    P = (Va*Ia + Vb*Ib + Vc*Ic)
    Q = (Ia*Vbc + Ib*Vca + Ic*Vab)
    Ssq = P**2 + Q**2
    R = P*Vm**2/Ssq
    X = Q*Vm**2/Ssq
    """
    
def model_machine():
    """
    may be better as a class
    Model a synchronous machine with inertia - Tau=1/2Jw^2
    inputs: voltage waveform, inertia, MVAbase
    outputs: shaft speed and shaft torque
    """
    
    # Write Logic to detect Switching Action
def trip_element():
    """
    Determine when to assert the trip based on sycn machine output
    inputs: shaft speed, shaft torque, thresholds
    outputs: boolean trip, counters
    """

def play_sound():
    print ("Playing sound")
        
    sd.default.samplerate = 44100
    # Generate time of samples between 0 and two seconds
    dur = 2.0
    frequency = 440    
    fs = 44100 # DAC sample rate
    samples = np.arange(fs * dur) / fs
    # Recall that a sinusoidal wave of frequency f has formula w(t) = A*sin(2*pi*f*t)
    wave = 2**15 * np.sin(2 * np.pi * frequency * samples) # full scale = 2^16 (peak to peak)
    # Convert it to wav format (16 bits)
    wav_wave = np.array(wave, dtype=np.int16)
    sd.play(wav_wave, blocking=True)

    print("Done playing sound")

def play_signal(Va):
    print ("Playing signal to sound card")
        
    sd.default.samplerate = 44100

    wave = 2**14 * Va # full scale = 2^16 (peak to peak)
    # Convert it to wav format (16 bits)
    wav_wave = np.array(wave, dtype=np.int16)
    sd.play(wav_wave, blocking=True)

    print("Done playing signal")

class Life:
    """Represents human life and all it can do
    
    Attributes:
        age: number of years old the life is
        alive: boolean indiacting if the life is still existance
    """
    def __init__(self):
        self.age = 0
        self.alive = True
    
    def growup(self,years):
        """Ages life by ``years``"""
        self.age += years
        if self.age > 100:
            self.die()
        
    def die(self):
        """Ends the life"""
        self.alive = False
    
def lifecycle():
    matt = Life()
    
    # <att goes through elementary school
    matt.growup(12)
    
    # keep growinng up till he dies
    while matt.alive:
        matt.growup(1)
        print("hes still kicking")
    
    print("rip in pieces")

def life():
    
    Va = make_wave()
    
    #decode_wave()
    #plt.show()
    #play_sound()
    play_signal(Va)
    

def the_circle_of_life():
    life()
    #lifecycle()

if __name__ == "__main__":
    the_circle_of_life()