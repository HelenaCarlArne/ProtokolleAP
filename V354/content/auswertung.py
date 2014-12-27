import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
print(80*'@')

#=======================#
# Laden der Gerätedaten #
#=======================#

L, L_err, C, C_err, R_1, R_1_err, R_2, R_2_err = np.loadtxt('../Werte/daten_bauteile.txt', unpack=True)

#==============================================================================================#
# Aufgabe (a) - Bestimmung des effektiven Dämpfungswiderstandes R_eff und der Abklingdauer T_x # <-- TODO: Plots müssen noch schön gemacht werden
#==============================================================================================#
print('Aufgabe a:')
print('~~~~~~~~~~')

# Lade Daten
t, U = np.loadtxt('../Werte/daten_a.txt', unpack=True)

# Daten verschieben <-- Damit der Plot symmetrisch zur x-Achse wird und bei x=0 beginnt
t -= t.min()
U -= 43

# Finde Maxima und Minima der gedämpften Schwingung <-- Werden auch gemeinsam in t_ext und U_ext gespeichert
import peakdetect_funktion
max, min = peakdetect_funktion.peakdetect(U, t, lookahead=100, delta=0)
max = np.array(max).T
min = np.array(min).T
t_ext = []
t_ext.extend(max[0,:])
t_ext.extend(min[0,:])
U_ext = []
U_ext.extend( max[1,:])
U_ext.extend(-min[1,:]) # <-- Minima werden an x-Achse gespiegelt, damit der Logarithmus keine Probleme bekommt

# Lineare Ausgleichsrechnung mit den logarithmierten Spannungen:
def f(x, m, b):
    return m*x + b
params_oE, cov_oE = curve_fit(f, max[0,:], np.log(abs(max[1,:]))) # <-- Nur für den Plot wichtig. Nicht ins Protokoll!
params_uE, cov_uE = curve_fit(f, min[0,:], np.log(abs(min[1,:]))) # <-- Nur für den Plot wichtig. Nicht ins Protokoll!
params, cov = curve_fit(f, t_ext, np.log(U_ext)) # <-- Für die Auswerung relevante Ausgleichsrechnung

R_eff     = -2*params[0]*L # <-- Folgt aus Gleichung (5) und der zweiten Gleichung auf Seite 287
R_eff_err = np.sqrt((-2*L*np.sqrt(cov[0][0]))**2+(-2*params[0]*L_err)**2) # <-- Wichtige Fehlerformel für den Auswertungsteil
T_x       = -1/params[0] # <-- Folgt aus Gleichung (5) und der zweiten Gleichung auf Seite 287
T_x_err   = np.sqrt((1/params[0]**2*np.sqrt(cov[0][0]))**2) # <-- Wichtige Fehlerformel für den Auswertungsteil

# Berechne T_x aus Gerätedaten
T_x_theo     = 2*L/R_1 # <-- Zweite Gleichung auf Seite 287
T_x_theo_err = np.sqrt((2/R_1*L_err)**2+(-2*L/R_1**2*R_1_err)**2) # <-- Wichtige Fehlerformel für den Auswertungsteil

# Plot der Schwingungskurve erstellen
x = np.linspace(0.00, 0.0005, 20000)
plt.plot(t, U, 'r-', label='Schwingungskurve')
plt.plot(max[0], max[1], 'kx', label='Extrema')
plt.plot(min[0], min[1], 'kx')
plt.plot(x,  np.exp(params_oE[0]*x+params_oE[1]), 'k-', label='Einhüllende')
plt.plot(x, -np.exp(params_uE[0]*x+params_uE[1]), 'k-')
plt.ylim(-80, 80)

#############################################################
######KOMMENTAAAR############################################
#### Maik ist wunderbar, manchmal###
#############################################################

plt.xlim(0, 0.000450)
plt.xlabel(r'$t\,[\mathrm{\mu s}]$')
plt.ylabel(r'$U_C\,[\mathrm{V}]$')
plt.xticks([0.0000, 0.0001, 0.0002, 0.0003, 0.0004],
           [r"$0$", r"$100$", r"$200$", r"$300$", r"$400$"])
plt.legend(loc = "best")
plt.tight_layout()
plt.savefig("../build/plot_schwingungskurve.pdf")
plt.show()
plt.close()

# Halblogarithmischen Plot der Einhüllenden erstellen
x = np.linspace(0.00, 0.0005, 20000)
plt.semilogy(t_ext, U_ext, 'rx', label='Messdaten der Extrema')
plt.plot(x, np.exp(params[0]*x+params[1]), 'k-', label='Regressionsgerade')
plt.xlim(t.min(), t.max())
plt.xlabel(r'$t\,[\mathrm{\mu s}]$')
plt.ylabel(r'$U_C$')
plt.xticks([0.0000, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005],
           [r"$0$", r"$100$", r"$200$", r"$300$", r"$400$", r"$500$"])
plt.legend(loc = "best")
plt.tight_layout()
#plt.show()
plt.savefig("../build/plot_einhuellende_semilog.pdf")
plt.close()


# Ergebnisse ausgeben
print('Maxima der Schwingungskurve:')
print('t[s]\t\tU_C[V]')
for i in range(0, len(max[0,:])):
    print('{:.6f}\t{:.2f}'.format(max[0,i], max[1,i]))
print('Minima der Schwingungskurve:')
print('t[s]\t\tU_C[V]')
for i in range(0, len(min[0,:])):
    print('{:.6f}\t{:.2f}'.format(min[0,i], min[1,i]))
print('Die lineare Ausgleichsrechnung ergibt die logarithmisierte Funktion:')
print('------------------------------------------------------------')
print('| ln(U_C(t)) = ({:.3f}+-{:.3f})1/s * t + ({:.3f}+-{:.3f}) |'.format(params[0], np.sqrt(cov[0][0]), params[1], np.sqrt(cov[1][1])))
print('------------------------------------------------------------')
print('Gemessener Dämpfungswiderstand R_eff:\t({:.2f}+-{:.2f}) Ohm'.format(R_eff, R_eff_err))
print('Verwendeter Widerstand R_1:\t\t({}+-{}) Ohm'.format(R_1, R_1_err))
print('Relative Abweichung:\t\t\t{:.2f} %'.format(abs(R_1-R_eff)/R_1*100))
print('Gemessene Abklingzeit:\t\t\t({:.6f}+-{:.6f}) s'.format(T_x, T_x_err))
print('Berechnete Abklingzeit:\t\t\t({:.6f}+-{:.6f}) s'.format(T_x_theo, T_x_theo_err))
print('Relative Abweichung:\t\t\t{:.2f} %'.format(abs(T_x_theo-T_x)/T_x_theo*100))
print(80*'@')



#=========================================================#
# Aufgabe (b) - Bestimmung des Dämpfungswiderstandes R_ap # <-- Soweit fertig. :D
#=========================================================#
print('Aufgabe b:')
print('~~~~~~~~~~')

# Lade Messdaten
R_ap = np.loadtxt('../Werte/daten_b.txt', unpack=True)

# Berechne R_ap aus Gerätedaten
R_ap_theo     = 2 * np.sqrt(L/C)
R_ap_theo_err = np.sqrt((np.sqrt(1/(L*C))*L_err)**2+(np.sqrt(L*C)*C_err)**2) # <-- Wichtige Fehlerformel für den Auswertungsteil

# Ergebnisse ausgeben
print('Gemessener R_ap:\t\t\t{} Ohm'.format(R_ap))
print('Berechneter R_ap:\t\t\t({:.2f}+-{:.2f}) Ohm'.format(R_ap_theo, R_ap_theo_err))
print('Relative Abweichung:\t\t\t{:.2f} %'.format(abs(R_ap_theo-R_ap)/R_ap_theo*100))
print(80*'@')


#=============================================================================#
# Aufgabe (c) - Untersuchung der Frequenzabhängigkeit der Kondensatorspannung # <-- TODO: Plots müssen noch schön gemacht werden
#=============================================================================#
print('Aufgabe c:')
print('~~~~~~~~~~')

# Lade Messdaten
f, U_C, U_0 = np.loadtxt('../Werte/daten_c.txt', unpack=True)

# Halblogarithmischen Plot der Messdaten erstellen
plt.semilogy(f, U_C/U_0, 'rx', label="Messdaten")
plt.xlabel(r'$f\,[\mathrm{kHz}]$')
plt.ylabel(r'$U_{\mathrm{C}}/U_{\mathrm{0}}$')
plt.xlim(f.min(), f.max())
plt.yticks([10e-2, 5*10e-2, 10e-1, 5*10e-1],
           [r"$10^{-1}$", r"$5\cdot 10^{-1}$", r"$1$", r"$5$"])
plt.xticks([10000, 15000, 20000, 25000, 30000, 35000, 40000, 45000, 50000],
           [r"$10$", r"$15$", r"$20$", r"$25$", r"$30$", r"$35$", r"$40$", r"$45$", r"$50$"])
plt.legend(loc = "best")
plt.tight_layout()
plt.savefig('../build/plot_amplitude_semilog.pdf')
plt.close()

# Doppelt-logarithmischen Plot der Messdaten erstellen <-- Sieht aber nicht so schön aus. Kann man auch weglassen!
'''plt.loglog(f, U_C/U_0, 'rx', label="Messdaten")
plt.xlabel(r'$f\,[\mathrm{Hz}]$')
plt.ylabel(r'$U_{\mathrm{C}}/U_{\mathrm{0}}$')
plt.xlim(f.min(), f.max())
#x-achsen Beschriftung beigegeben
plt.xticks([10e3, 2*10e3, 3*10e3, 4*10e3, 5*10e3],
           [r"$10^{4}$", r"$2\cdot 10^{4}$", r"$3\cdot 10^{4}$", r"$4\cdot 10^{4}$", r"$5\cdot 10^{4}$"])
plt.legend(loc = "best")
plt.tight_layout()
plt.savefig('../build/plot_amplitude_log.pdf')
plt.close()'''

# Berechnen der Resonanzüberhöhung (Güte q) und der Breite der Resonanzkurve
resonanzkurve = np.where(U_C > U_C.max()/np.sqrt(2))
breite        = f[resonanzkurve].max() - f[resonanzkurve].min()
q             = U_C.max()/28.8

# Berechnen von q und der Breite der Resonanzkurve aus den Gerätedaten
breite_theo     = R_2/(L*2*np.pi) # <-- Gleichung (16). Umrechnung aus Kreisfrequenz beachten!
breite_theo_err = np.sqrt((1/(L*2*np.pi)*R_2_err)**2+(-R_2/(L**2*2*np.pi)*L_err)**2) # <-- Wichtige Fehlerformel für den Auswertungsteil
q_theo          = 1/R_2 * np.sqrt(L/C) # <-- Gleichung (15)
q_theo_err      = np.sqrt((-1/(R_2**2)*np.sqrt(L/C)*R_2_err)**2+(1/(2*C*R_2)*np.sqrt((C/L)*L_err))**2+(-L/(C**2*2*R_2)*np.sqrt(C/L)*C_err)**2) # <-- Wichtige Fehlerformel für den Auswertungsteil

# Linearen Plot im Bereich der Resonanzfrequenz erstellen
plt.plot(f[resonanzkurve], U_C[resonanzkurve]/U_0[resonanzkurve], 'rx', label="Messdaten")
plt.axvline(f[resonanzkurve].min(), color='g', linestyle='--', label="$f_-$")
plt.axvline(f[resonanzkurve].max(), linestyle='--', label="$f_+$")
plt.xlim(0.95*f[resonanzkurve].min(), 1.05*f[resonanzkurve].max())
plt.ylim(0.95*(U_C[resonanzkurve]/U_0[resonanzkurve]).min(), 1.05*(U_C[resonanzkurve]/U_0[resonanzkurve]).max())
plt.xlabel(r'$f\,[\mathrm{kHz}]$')
plt.ylabel(r'$U_{\mathrm{C}}/U_{\mathrm{0}}$')
plt.xticks([22000, 24000, 26000, 28000, 30000],
           [r"$22$", r"$24$", r"$26$", r"$28$", r"$30$"])
plt.legend(loc = "best")
plt.tight_layout()
plt.savefig('../build/plot_amplitude_linear.pdf')
plt.close()

# Ergebnisse ausgeben
print('Gemessene Maximalspannung U_C:\t\t{:.1f} V'.format(U_C.max()))
print('Gemessenes f_p:\t\t\t\t{:.1f} Hz'.format(f[resonanzkurve].max()))
print('Gemessenes f_m:\t\t\t\t{:.1f} Hz'.format(f[resonanzkurve].min()))
print('Gemessene Resonanzüberhöhung q:\t\t{:.3f}'.format(q))
print('Berechnete Resonanzüberhöhung q:\t{:.2f}+-{:.2f}'.format(q_theo, q_theo_err))
print('Relative Abweichung:\t\t\t{:.2f} %'.format(abs(q_theo-q)/q_theo*100))
print('Gemessene Resonanzbreite:\t\t{:.2f} Hz'.format(breite))
print('Berechnete Resonanzbreite:\t\t({:.2f}+-{:.2f}) Hz'.format(breite_theo, breite_theo_err))
print('Relative Abweichung:\t\t\t{:.2f} %'.format(abs(breite_theo-breite)/breite_theo*100))
print(80*'@')

#===========================================================#
# Aufgabe (d) - Frequenzabhängigkeit der Phasenverschiebung #
#===========================================================#
########################################################################
# Teil d)
#nu, deltat = np.loadtxt('messwerte_d.txt', unpack=True)

#phi = 2 * np.pi * deltat * 1e-9 * nu * 1e3

#plt.subplot(211)
#plt.semilogy(nu, phi, '+')
#plt.grid(True, which='both')
#plt.xlabel('$\\nu/\\mathrm{kHz}$')
#plt.ylabel('$\\phi$')

#sec = np.where(nu > 25)

#plt.subplot(212)
#plt.grid(True, which='both')
#plt.xlabel('$\\nu/\\mathrm{kHz}$')
#plt.ylabel('$\\phi$')
#plt.plot(nu[sec], phi[sec], '+')

#plt.title('Phase gegen Frequenz')
#plt.savefig('phasen-plot.pdf')
#plt.close()

# Daten laden und umrechnen
f_phase, delta_t = np.loadtxt('../Werte/daten_d.txt', unpack = True)
# [Hz]      [µs]
delta_t /= 1000000 # in s umrechnen
phi=f_phase*delta_t*2*np.pi

# Plotten
x = np.linspace(0.01, 5, 20000)
plt.semilogx(f_phase, phi, 'rx', label = 'Messdaten')
plt.xlabel(r'$f\,[\mathrm{Hz}]$')
plt.ylabel(r'$\varphi$')
plt.xlim(f_phase.min(), f_phase.max())
#x-achsen Beschriftung beigegeben
plt.xticks([10e3, 2*10e3, 3*10e3, 4*10e3, 5*10e3],
           [r"$10^{4}$", r"$2\cdot 10^{4}$", r"$3\cdot 10^{4}$", r"$4\cdot 10^{4}$", r"$5\cdot 10^{4}$"])
plt.legend(loc = "lower right")
plt.tight_layout()
plt.savefig("../build/plot_phase.pdf")
plt.close()

#linearer Teil um pi/2
plt.plot(f_phase, phi, 'rx', label="Messdaten")
plt.axvline(22e3, color='g', linestyle='--', label="$f_1$")
plt.axvline(30e3, linestyle='--', label="$f_2$")
plt.xlim(0.95*22e3, 1.05*30e3)
plt.ylim(0, 3)
plt.xlabel(r'$f\,[\mathrm{kHz}]$')
plt.ylabel(r'$\phi$')
plt.xticks([22000, 24000, 26000, 28000, 30000],
           [r"$22$", r"$24$", r"$26$", r"$28$", r"$30$"])
plt.legend(loc = "best")
plt.tight_layout()
plt.savefig('../build/plot_phase_linear.pdf')
plt.close()

print('f_res:\t{:.2f} kHz'.format(26.0))
