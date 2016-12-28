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

def hello_world():
    print ("Setting Up Waveforms")
    
    t = np.arange(0,2,0.001)
    len = np.size(t,0)
        
    ampl = np.linspace(1,2,len)
    freq = np.linspace(1,1,len)
    
    # model contingency deviation
    
    # include magnitude and phase noise
    noiseA = np.random.uniform(0,1,len)
    noiseF = np.random.uniform(0,1,len)
    
    # Generate waveforms    
    phase = 2*np.pi*freq*t
    a = ampl*np.sin(2*np.pi*freq*t)
    b = ampl*np.sin(2*np.pi*freq*t + 2*np.pi/3)
    c = ampl*np.sin(2*np.pi*freq*t - 2*np.pi/3)
       
    
    # Plot
    plt.close('all')
    fig1 = plt.figure(1,figsize=(3*1.6,3*1.6))
    fig1.clf()
    plt.plot(t,a)
    plt.plot(t,b,'r')
    plt.plot(t,c,'g')
    plt.grid()
    
    
    fig2 = plt.figure(2,figsize=(3*1.6,3*1.6))
    fig2.clf()
    plt.plot(t,noiseA)
    plt.plot(t,noiseF,'k')
    plt.grid()
    #plt.show()


    
    print("waveforms ready")

def goodbye_world():
    print ("Start Decoding Waveforms")
    
    # Calculate RMS quantities
    
    # Model Sync machine to determine signals
    
    # Write Logic to detect Switching Action

def play_output():
    print ("Output to sound card")
        
    sd.default.samplerate = 44100
    # Generate time of samples between 0 and two seconds
    time = 2.0
    frequency = 440    
    fs = 44100 # DAC sample rate
    samples = np.arange(fs * time) / fs
    # Recall that a sinusoidal wave of frequency f has formula w(t) = A*sin(2*pi*f*t)
    wave = 2**15 * np.sin(2 * np.pi * frequency * samples) # full scale = 2^16 (peak to peak)
    # Convert it to wav format (16 bits)
    wav_wave = np.array(wave, dtype=np.int16)
    sd.play(wav_wave, blocking=True)

    print("Done playing sound")

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
    #hello_world()
    #goodbye_world()
    #plt.show()
    play_output()
    

def the_circle_of_life():
    life()
    #lifecycle()

if __name__ == "__main__":
    the_circle_of_life()