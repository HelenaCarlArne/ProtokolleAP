import os.path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
from scipy.optimize import curve_fit
from matrix2latex import matrix2latex

# Apparate- und Naturkonstanten
d = 0.0076250   # [m]
Δd = 0.0000051  # [m]
ρ_L = 0         # näherungsweise
ρ_Oel = 886     # [kg/m^3]
g = const.g     # [m/s^2]
B = 8.266e-3    # [Pa/m]
p = 1e5         # [Pa]
s = 0.0005      # [m]

if((os.path.exists('../Werte/daten_tröpfchen_liste.txt'))-1): 

    # Beim erstmaligen Ausführen des Auswertungsskriptes wird eine Datei mit den
    # berechneten Geschwindigkeiten erstellt, die es ermöglicht einzelne Tröpfchen
    # von der folgenden Berechnung auszuschließen.

    Tröpfchen = 25 # Anzahl der untersuchten Tröpfchen eingeben.
    
    v_auf = np.zeros(Tröpfchen)
    Δv_auf = np.zeros(Tröpfchen)
    v_ab = np.zeros(Tröpfchen)
    Δv_ab = np.zeros(Tröpfchen)   
    
    U, t_0, T = np.loadtxt('../Werte/daten_tröpfchen.txt', unpack = True) # [V],[s],[K]
    v_0 = s/t_0
    
    teilchen_liste = open('../Werte/daten_tröpfchen_liste.txt', 'w')
    teilchen_liste.write('# Nr    U    v_auf    dv_auf    v_ab    dv_ab    v_0    T    OK\n')
    
    for i in range(0,Tröpfchen):
        t_auf, t_ab = np.loadtxt('../Werte/daten_tröpfchen_{}.txt'.format(i+1), unpack = True) # [s],[s]
        n = len(t_auf)
        v_auf[i] = np.mean(s/t_auf)
        Δv_auf[i] = np.sqrt(1/(n*(n-1))*sum((s/t_auf-np.mean(s/t_auf))**2))
        v_ab[i] = np.mean(s/t_ab)
        Δv_ab[i] = np.sqrt(1/(n*(n-1))*sum((s/t_ab-np.mean(s/t_ab))**2))
        teilchen_liste.write('{}    {:.0f}    {:.16f}    {:.16f}    {:.16f}    {:.16f}    {:.16f}    {}    1\n'.format(i+1, U[i], v_auf[i], Δv_auf[i], v_ab[i], Δv_ab[i], v_0[i], T[i]))
        print('Teilchen {}: {:.5f}'.format(i+1, abs((2*v_0[i]-v_ab[i]+v_auf[i])*1000)))
    
    teilchen_liste.close()
    
else:

    # Daten laden
    Nr, U, v_auf, Δv_auf, v_ab, Δv_ab, v_0, T, OK = np.loadtxt('../Werte/daten_tröpfchen_liste.txt', unpack = True)

    # Anzahl aller Tröpfchen
    n = len(Nr)

    # Viskosität der Luft (unkorrigiert)
    η_L = 47.0588e-9*T+4.44329e-6 # Aus Zeichnung in der Versuchsanleitung abgeleitet
    
    # Tröpfchenradius
    r = np.sqrt((9*η_L*(v_ab-v_auf))/(2*g*(ρ_Oel-ρ_L))) 
    Δr = np.sqrt((9*η_L*(Δv_ab**2+Δv_auf**2))/(8*g*(ρ_Oel-ρ_L)*(v_ab-v_auf)))

    # Elektrisches Feld
    E = U/d
    ΔE = U/d**2*Δd

    # Tröpfchenladung (unkorrigiert)
    q = 3*np.pi*η_L*np.sqrt((9*η_L*(v_ab-v_auf))/(4*g*(ρ_Oel-ρ_L)))*((v_ab+v_auf)/E)
    Δq = 3*np.pi*η_L*np.sqrt((9*η_L)/(4*g*(ρ_Oel-ρ_L)))/E*np.sqrt(((3*v_ab**2+2*v_ab*v_auf-v_auf**2)/(np.sqrt((v_ab-v_auf)*(v_ab+v_auf)**2))*Δv_ab)**2+((v_ab**2-2*v_ab*v_auf-3*v_auf**2)/(np.sqrt((v_ab-v_auf)*(v_ab+v_auf)**2))*Δv_auf)**2+(np.sqrt((v_ab-v_auf)*(v_ab+v_auf)**2)/E*ΔE)**2) # Ächz

    # Viskosität der Luft (korrigiert)
    η_eff = η_L*(1/(1+B/(p*r)))
    Δη_eff = η_L*(B*p)/((B+p*r)**2)*Δr

    # Tröpfchenladung (korrigiert)
    q_korr = q*(np.sqrt(1+B/(p*r)))**3
    Δq_korr = np.sqrt((1+B/(p*r))**3*Δq**2+q**2*(9*B**2*(B+p*r))/(4*p**3*r**5)*Δr**2)
   
######### Debug-Ausgabe ##############
    #print('v_auf')
    #print(v_auf)
    #print('v_ab')
    #print(v_ab)
    print('E')
    print(E)
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print('')
    print('eta')
    print(η_L)

    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print('')
    print('r')
    print(r)
    print(Δr)

    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print('')
    print('q')
    print(q)
    print(Δq)

    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print('')
    print('eta')
    print(η_eff)
    print(Δη_eff)

    print('@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print('')
    print('q_korr')
    print(q_korr)
    print(Δq_korr)
     
    # Bestimmung des bestmöglichen ggTs mit Plot 
    x = np.linspace(1.00e-19,1.8e-19,16000)
    y = np.zeros(16000)
    for i in range(0,len(x)):
        for j in range(0, len(q_korr)):
            y[i] += abs(1-(q_korr[j]/(round(q_korr[j]/x[i])))/x[i])
        y[i] /= len(q_korr)
    z = y.argmin()   
    plt.plot(x*1e19, y, 'k-')   
    plt.axvline(1.605, linestyle='--', color='green', label='Literaturwert')
    plt.axvline(x[z]*1e19, linestyle='--', color='blue', label='Bestmöglicher ggT')
    plt.xlim(1.00, 1.8)
    plt.xlabel(r"$\mathrm{Ladung}\; q /10^{-19}\mathrm{C}$")
    plt.ylabel(r'Mittlerer relativer Fehler aller Ladungen')
    plt.legend(loc = 'best')
    plt.savefig("../Bilder/plot_ggT.pdf")
    plt.close()   
    
    print('Bestmöglicher ggT der errechneten Ladungen:')
    print('e = {} C'.format(x[z]))
    
    # Graphen plotten
    o = np.linspace(1,27,25)
    plt.errorbar(r[0]*1e6, q_korr[0]*1e19, xerr=Δr[0]*1e6, yerr=Δq_korr[0]*1e19, fmt='rx', label='Messwerte')
    for j in range(1, len(q_korr)):
        if(OK[j]):
            plt.errorbar(r[j]*10**6, q_korr[j]*10**19, xerr=Δr[j]*10**6, yerr=Δq_korr[j]*10**19, fmt='rx') # Sollen die 
    plt.legend(loc = 'best')
    plt.ylim(0, 20)
    plt.xlim(0,1.4)
    plt.xlabel(r"$\mathrm{Ladung}\; q /10^{-19}\mathrm{C}$")
    plt.ylabel(r"$\mathrm{Tröpfchenradius}\; r /\mathrm{\mu m}$")
    plt.savefig("../Bilder/plot_messwerte.pdf")
    plt.close()
    
    o = np.linspace(1,27,25)
    plt.errorbar(r[0]*1e6, q_korr[0]*1e19, xerr=Δr[0]*1e6, yerr=Δq_korr[0]*1e19, fmt='rx', label='Messwerte')
    for j in range(1, len(q_korr)):
        if(OK[j]):
            plt.errorbar(r[j]*1e6, q_korr[j]*1e19, xerr=Δr[j]*1e6, yerr=Δq_korr[j]*1e19, fmt='rx') # Sollen die Fehler auch geplottet werden? Sie sind so klein.
            #plt.plot(r[j], q_korr[j], 'rx')
            #plt.plot(r[j]*1e9, q_korr[j], 'rx')
    

    for j in range(0, 30):
        plt.axhline(j*x[z]*1e19, linestyle='--', color='grey') # Literaturwerte
        
    plt.legend(loc = 'best')
    plt.ylim(0, 6)
    plt.xlabel(r"$\mathrm{Ladung}\; q /10^{-19}\mathrm{C}$")
    plt.ylabel(r"$\mathrm{Tröpfchenradius}\; r /\mathrm{\mu m}$")
    plt.savefig("../Bilder/plot_messwerte+.pdf")
    plt.close()
