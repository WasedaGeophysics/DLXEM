from script.emulatte.forwardscr import emgmodel
from script.emulatte.forwardscr.transmitter import *

def model(thicks, **kwargs):
    mdl = emgmodel.Subsurface1D(thicks, **kwargs)
    return mdl

def transmitter(name, freqtime, **kwargs):
    cls = globals()[name]
    tmr = cls(freqtime, **kwargs)
    return tmr