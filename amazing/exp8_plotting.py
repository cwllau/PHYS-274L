import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy import asarray as ar, exp
from pylab import rcParams

rcParams['figure.figsize'] = 18, 9

## Main Directory
main_dir = "/Volumes/FLASH/PHYS274L/"
plt.clf()

## Constants:
c = 299792458.0 # speed of light
h = 6.62607004e-34
Beta = c * h
m_e = 9.10938356e-31 #mass of electron in kg
c_e = 1.60217662e-19 #chage of electron in C
h_bar = h / (2*np.pi) #reduced planks constant 
J_ev = 1.60218e-19 # convert eV to J
	#B = m_e * E / (c_e * h_bar)
gamma = m_e / (c_e * h_bar)


	##Fit gaussian 589.382 - 590.02
D2_start = 589.3
D2_end = 590.625
D1_start = 588.38
D1_end = 590.0

def gaus(x,a,x0,sigma):
	return a*np.exp(-(x-x0)**2/(2*sigma**2))

## Set Axis Labels
plt.ion()
plt.xlabel("Lambda [nm]")
plt.ylabel("Intensity")

## Loading Data:
sodium_L = np.genfromtxt(main_dir+"sample_data.dat", delimiter=',', usecols=(0))
sodium_I = np.genfromtxt(main_dir+"sample_data.dat", delimiter=',', usecols=(1))
#plt.plot(sodium_L, sodium_I, 'b', label="Raw Data", alpha=0.7)

#instrument_L = np.genfromtxt(main_dir+"/background_instrument.dat", delimiter=',', usecols=(0))
instrument_I = np.genfromtxt(main_dir+"background_instrument.dat", delimiter=',', usecols=(1))

#plt.plot(sodium_L, instrument_I, 'r', label="Instrument Error", alpha=0.7)

#atmos_L = np.genfromtxt(main_dir+"/background_atmosphere.dat", delimiter=',', usecols=(0))
atmos_I = np.genfromtxt(main_dir+"background_atmosphere.dat", delimiter=',', usecols=(1))

#plt.plot(sodium_L,atmos_I, 'g', label="Background Error", alpha=0.7)


## Correct the spectrum
corr_I = sodium_I - sum(instrument_I+atmos_I)/(len(instrument_I))
corr_L = sodium_L - (sodium_L[np.argmax(sodium_I)] - 588.995)
corr_I_err = (sum(instrument_I+atmos_I))/(len(instrument_I))
corr_L_err = (sodium_L[np.argmax(sodium_I)] - 588.995)

plt.plot(corr_L, corr_I, 'k.', label="Corrected Data", linewidth=1, alpha=1)
#plt.errorbar(corr_L, corr_I, yerr=(sum(instrument_I+atmos_I))/(len(instrument_I)), fmt='o')
#plt.legend()
#plt.show()

D2_index = []
D1_index = []


## Selecting the data points belonging to the D1/D2 spectra
for i in range(len(corr_L)):
	if corr_L[i] >= D2_start and corr_L[i] <= D2_end:
		D2_index.append(i)
	elif corr_L[i] >= D1_start and corr_L[i] <= D1_end:
		D1_index.append(i)

D2_L = np.array([corr_L[m] for m in D2_index])
D2_I = np.array([corr_I[n] for n in D2_index])

#plt.legend()
#plt.show()

'''
Delta_E2 = (-12400.0)*(D1_L[np.argmax(D1_I)]-D2_L[np.argmax(D2_I)])*10.0/(D1_L[np.argmax(D1_I)]*10.0)**2
print("Delta E = "+str(Delta_E2)+" [eV]")

B_mag2 = gamma * Delta_E2 * J_ev 
print("|B|= "+str(B_mag2)+" [T]")
'''


## Fitting Gaussians to lines
N = len(D2_L)

## Fit for D2
mean_D2 = sum(D2_L*D2_I)/sum(D2_I)
sigma_D2 = np.sqrt(sum(D2_I*(D2_L - mean_D2)**2)/sum(D2_I))

popt, pcov = curve_fit(gaus, D2_L, D2_I, p0=[max(D2_I),mean_D2,sigma_D2], sigma=corr_I_err)

#plt.plot(D2_L, D2_I, 'k', label="Data")
plt.plot(D2_L, gaus(D2_L, *popt),'m', linewidth=2, label='D2 fit')

## Fit for D1
mean_D1 = sum(D1_L*D1_I)/sum(D1_I)
sigma_D1 = np.sqrt(sum(D1_I*(D1_L - mean_D1)**2)/sum(D1_I))

popt2, pcov2 = curve_fit(gaus, D1_L, D1_I, p0=[max(D1_I),mean_D1,sigma_D1], sigma=corr_I_err)

#plt.plot(D2_L, D2_I, 'k', label="Data")
plt.plot(D1_L, gaus(D1_L, *popt2),'r', linewidth=2, label='D1 fit')

plt.plot(D2_L, D2_I, 'k.', ms=10, label="D2 Line")

'''
## -- Calculations with peark values

print("Peak for D2: "+str(D2_L[np.argmax(D2_I)])+" [nm]")
E2 = Beta / (D2_L[np.argmax(D2_I)]*1e-9)
print("D2 Energy: "+str(E2)+" [J]")
plt.annotate(("D2 peak: "+str(D2_L[np.argmax(D2_I)])+', '+str(max(D2_I))),(D2_L[np.argmax(D2_I)], max(D2_I)))


D1_L = np.array([corr_L[k] for k in D1_index])
D1_I = np.array([corr_I[z] for z in D1_index])
plt.plot(D1_L, D1_I, 'k.', ms=10, label="D1 Line")

print("Peak for D1: "+str(D1_L[np.argmax(D1_I)])+" [nm]")
E1 = Beta / (D1_L[np.argmax(D1_I)]*1e-9)
print("D1 Energy: "+str(E1)+" [J]")
plt.annotate(("D1 peak: "+str(D1_L[np.argmax(D1_I)])+', '+str(max(D1_I))),(D1_L[np.argmax(D1_I)], max(D1_I)))

Delta_E = abs(E2 - E1)
delta_E = h*c*np.sqrt(((0.1*1e-9)/(D1_L[np.argmax(D1_I)]*1e-9)**2)**2+((0.1*1e-9)/(D2_L[np.argmax(D2_I)]*1e-9)**2)**2)
print("Energy Difference: "+str(Delta_E)+" +/- "+str(delta_E)+" [J]")

B_mag = gamma * Delta_E
delta_B  = gamma * delta_E
print("|B| = "+str(B_mag)+" +/- "+str(delta_B)+" [T]")
'''


## -- Calcuations with fitted values

print("Peak for D2: "+str(589.675)+" [nm]")
E2 = Beta / (589.675*1e-9)
print("D2 Energy: "+str(E2)+" [J]")
plt.annotate(("D2 peak: "+str(589.675)+', '+str(176.404)),(589.675, 176.404))


D1_L = np.array([corr_L[k] for k in D1_index])
D1_I = np.array([corr_I[z] for z in D1_index])
plt.plot(D1_L, D1_I, 'k.', ms=10, label="D1 Line")

print("Peak for D1: "+str(589.015)+" [nm]")
E1 = Beta / (589.015*1e-9)
print("D1 Energy: "+str(E1)+" [J]")
plt.annotate(("D1 peak: "+str(589.015)+', '+str(314.639)),(589.05, 314.639))

Delta_E = abs(E2 - E1)
delta_E = h*c*np.sqrt(((0.00500799*1e-9)/(589.015*1e-9)**2)**2+((0.01133*1e-9)/(589.675*1e-9)**2)**2)
print("Energy Difference: "+str(Delta_E)+" +/- "+str(delta_E)+" [J]")

B_mag = gamma * Delta_E
delta_B  = gamma * delta_E
print("|B| = "+str(B_mag)+" +/- "+str(delta_B)+" [T]")
'''
plt.xlim(588.3, 590.7)
plt.ylim(-20.0, 380.0)
plt.legend(loc=1)

