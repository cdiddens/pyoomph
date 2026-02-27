from pyoomph import *
from pyoomph.equations.harmonic_oscillator import HarmonicOscillator
from pyoomph.expressions import pi

eqs=HarmonicOscillator(omega=1)+InitialCondition(y=1)+ODEFileOutput()
p=Problem()
p+=eqs@"ho"
p.run(2*pi,startstep=0.1)