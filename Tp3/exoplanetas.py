# %%
import pyvo as vo
import matplotlib.pyplot as plt
import scipy.stats as sp
import scipy.special as spc
import numpy as np
import pandas as pd

# %%
service = vo.dal.TAPService("http://voparis-tap-planeto.obspm.fr/tap") #Elegimos de donde sacar los datos
query = """SELECT target_name,mass,period,star_spec_type, star_mass, detection_type, star_distance, semi_major_axis, radius, star_age
	FROM exoplanet.epn_core""" #Elegimos los datos que queremos.
#Los convertimos en un array
datos = service.search(query)
#Los convertimos en un dataframe para trabajar con pandas.
datos = datos.to_table().to_pandas()

# %%
# Ahora, separamos los planetas según su método de detección

velrad_check = datos.loc[:, 'detection_type'] == 'Radial velocity' #Filtramos los planetas que han sido detectados por velocidad radial
velrad = datos.loc[velrad_check] #Los separamos en un nuevo data frame.

#Y así con los demás métodos de detección

transit_check = datos.loc[:, 'detection_type'] == 'Primary Transit'
transit = datos.loc[transit_check]

imagen_check = datos.loc[:, 'detection_type'] == 'Imaging'
imagen = datos.loc[imagen_check]

microlens_check = datos.loc[:, 'detection_type'] == 'Microlensing'
microlens = datos.loc[microlens_check]

pulsar_check = datos.loc[:, 'detection_type'] == 'Pulsar'
pulsar = datos.loc[pulsar_check]

astrometry_check = datos.loc[:, 'detection_type'] == 'Astrometry'
astrometria = datos.loc[astrometry_check]

timing_check = datos.loc[:, 'detection_type'] == 'Timing'
timing = datos.loc[timing_check]

TTV_check = datos.loc[:, 'detection_type'] == 'TTV'
TTV = datos.loc[TTV_check]

otros_check = datos.loc[:, 'detection_type'] == 'Other'
otros = datos.loc[otros_check]

#Graficamos los datos

plt.plot(velrad['mass'], velrad['period'], 'ro', markersize=4, label='Velocidad radial')  
plt.plot(transit['mass'], transit['period'], '+', markersize=4, label='Tránsito')       
plt.plot(imagen['mass'], imagen['period'], 'g^', markersize=4, label='Imagen')           
plt.plot(microlens['mass'], microlens['period'], 'yv', markersize=4, label='Microlente') 
plt.plot(pulsar['mass'], pulsar['period'], 'mp', markersize=4, label='Púlsar')           
plt.plot(astrometria['mass'], astrometria['period'], 'cD', markersize=4, label='Astrometría')  
plt.plot(timing['mass'], timing['period'], 'k*', markersize=4, label='Timing')           
plt.plot(TTV['mass'], TTV['period'], 'wX', markersize=4, label='TTV', markeredgecolor='black') 
plt.plot(otros['mass'], otros['period'], 'o', color='purple', markersize=4, label='Otros')     
plt.xlabel('Masa (M$_\mathrm{J}$)')
plt.ylabel('Período (días)')
plt.xscale('log')
plt.yscale('log')
plt.legend(loc='upper left', fontsize='small')
plt.title('Masa vs Período según el método de detección')
plt.show()


#Un gráfico sin escala logarítmica en x pero recortando para 14 masas de Júpiter

plt.plot(velrad['mass'], velrad['period'], 'ro', markersize=4, label='Velocidad radial')  
plt.plot(transit['mass'], transit['period'], '+', markersize=4, label='Tránsito')       
plt.plot(imagen['mass'], imagen['period'], 'g^', markersize=4, label='Imagen')           
plt.plot(microlens['mass'], microlens['period'], 'yv', markersize=4, label='Microlente') 
plt.plot(pulsar['mass'], pulsar['period'], 'mp', markersize=4, label='Púlsar')           
plt.plot(astrometria['mass'], astrometria['period'], 'cD', markersize=4, label='Astrometría')  
plt.plot(timing['mass'], timing['period'], 'k*', markersize=4, label='Timing')           
plt.plot(TTV['mass'], TTV['period'], 'wX', markersize=4, label='TTV', markeredgecolor='black') 
plt.plot(otros['mass'], otros['period'], 'o', color='purple', markersize=4, label='Otros')     
plt.xlabel('Masa (M$_\mathrm{J}$)')
plt.ylabel('Período (días)')
plt.xlim(0.0001, 14)
plt.xscale('log')
plt.yscale('log')
plt.legend(loc='upper left', fontsize='small')
plt.title('Masa vs Período según el método de detección')
plt.show()


plt.hist(np.log(velrad['period']),bins = 'auto',density = True,histtype ='step', label = 'Velocidad radial',lw = 2)
plt.hist(np.log(transit['period']),bins = 'auto',density = True,histtype ='step', label = 'Tránsito',lw = 2)
plt.hist(np.log(imagen['period']),bins = 'auto',density = True,histtype ='step', label = 'Imagen',lw =2)
plt.hist(np.log(microlens['period']),bins = 'auto',density = True,histtype ='step', label = 'Microlentes', lw = 2)
plt.hist(np.log(pulsar['period']),bins = 'auto',density = True,histtype ='step', label = 'Pulsar', lw = 2)
plt.hist(np.log(astrometria['period']),bins = 'auto',density = True,histtype ='step', label = 'Astrometría',lw = 2)
plt.hist(np.log(timing['period']),bins = 'auto',density = True,histtype ='step', label = 'Timing',lw = 2)
plt.hist(np.log(TTV['period']),bins = 'auto',density = True,histtype ='step', label = 'TTV',lw = 2)
plt.hist(np.log(otros['period']),bins = 'auto',density = True,histtype ='step',label = 'Otros',lw = 2)
plt.legend()
plt.title('Distibucion de periodos para cada tecnica de deteccion')
plt.xlabel('log(Periodo) [Dias]')
plt.ylabel('Frecuencia', fontsize=14)
plt.show()



plt.hist(np.log(velrad['mass']), bins='auto', density=True, histtype='step', label='Velocidad radial', lw=2)
plt.hist(np.log(transit['mass']), bins='auto', density=True, histtype='step', label='Tránsito', lw=2)
plt.hist(np.log(imagen['mass']), bins='auto', density=True, histtype='step', label='Imagen', lw=2)
plt.hist(np.log(microlens['mass']), bins='auto', density=True, histtype='step', label='Microlentes', lw=2)
plt.hist(np.log(pulsar['mass']), bins='auto', density=True, histtype='step', label='Pulsar', lw=2)
plt.hist(np.log(astrometria['mass']), bins='auto', density=True, histtype='step', label='Astrometría', lw=2)
plt.hist(np.log(timing['mass']), bins='auto', density=True, histtype='step', label='Timing', lw=2)
plt.hist(np.log(TTV['mass']), bins='auto', density=True, histtype='step', label='TTV', lw=2)
plt.hist(np.log(otros['mass']), bins='auto', density=True, histtype='step', label='Otros', lw=2)
plt.legend()
plt.title('Distribución de masas para cada técnica de detección')
plt.xlabel('log(Masa) [M$_\mathrm{J}$]')
plt.ylabel('Frecuencia', fontsize=14)
plt.show()

# %% [markdown]
# Ejercicio 3

# %%
distancia = datos['star_distance']
plt.hist(distancia, color='blue', edgecolor='black', bins='auto')
plt.title('Distribución de distancias a las estrellas')
plt.xlabel('Distancia (pc)')
plt.ylabel('Frecuencia')
plt.show()

# %%
distancia = datos['star_distance']
plt.hist(distancia, color='blue', edgecolor='black', bins=np.logspace(np.log10(distancia.min()), np.log10(distancia.max()), 50))
plt.xscale('log')
plt.title('Distribución de distancias a las estrellas')
plt.xlabel('Distancia (pc)')
plt.ylabel('Frecuencia')
plt.xlim(1, 2000)
plt.show()

# %%
bins = np.logspace(np.log10(datos['star_distance'].min()), np.log10(datos['star_distance'].max()), 50)

#plt.hist(velrad['star_distance'], bins=bins, density=True, histtype='step', label='Velocidad radial', lw=2)
plt.hist(transit['star_distance'], bins=bins, density=True, histtype='step', label='Tránsito', lw=2)
plt.hist(imagen['star_distance'], bins=bins, density=True, histtype='step', label='Imagen', lw=2)
plt.hist(microlens['star_distance'], bins=bins, density=True, histtype='step', label='Microlentes', lw=2)
plt.hist(pulsar['star_distance'], bins=bins, density=True, histtype='step', label='Pulsar', lw=2)
plt.hist(astrometria['star_distance'], bins=bins, density=True, histtype='step', label='Astrometría', lw=2)
plt.hist(timing['star_distance'], bins=bins, density=True, histtype='step', label='Timing', lw=2)
plt.hist(TTV['star_distance'], bins=bins, density=True, histtype='step', label='TTV', lw=2)
#plt.hist(otros['star_distance'], bins=bins, density=True, histtype='step', label='Otros', lw=2)
plt.xlim(5, 2000)
plt.xscale('log')
plt.legend()
plt.title('Distribución de distancias a las estrellas para cada técnica de detección')
plt.xlabel('log(Distancia) [pc]')
plt.ylabel('Frecuencia', fontsize=14)
plt.show()

# %% [markdown]
# Realizamos un test KS para compararlo con una distribución normal y luego una exponencial

# %%
distance_check = datos.loc[:, 'star_distance'] < 2000
distance_filtered = datos.loc[distance_check]

va = distance_filtered['star_distance']
n = len(va)
#Ejes Y de las acumuladas, empezando desde 1 o desde 0
Smas = np.arange(1, n+1)/n
Smen = np.arange(0,	n)/n
teog = sp.norm.cdf(va, np.mean(va), np.std(va))
Des = np.append(teog - Smen, Smas - teog)
D = max(abs(Des))
test = D *	np.sqrt(n)
alpha = spc.kolmogi(0.05)
print('El test para una distribución normal es:', test, 'pero el valor crítico es:', alpha, 'por lo tanto no es una Gaussiana')

#Ahora, con una exponencial:

teoexp =	sp.expon.cdf(va,np.mean(va))

Desexp = np.append(teoexp - Smen, Smas - teoexp)
Dexp = max(abs(Desexp))
testexp = Dexp * np.sqrt(n)
print('El test para una distribución exponencial es:', testexp, 'pero el valor crítico es:', alpha, 'por lo tanto no es una exponencial')


# %% [markdown]
# Ejercicio 4

# %%
r = datos['radius']
m =	datos['mass']
plt.plot(r, m, '+', markersize=2)
plt.xlabel('Radio (R$_\mathrm{J}$)')
plt.ylabel('Masa (M$_\mathrm{J}$)')
plt.xscale('log')
plt.yscale('log')
plt.title('Masa vs Radio')
plt.show()

# %% [markdown]
# Tiene pinta de cúbica, probemos eso:
# 
# $y = log(M)$
# 
# $x = log(R)$
# 
# Propongo:
# 
# $y = ax^3+bx^2+c$

# %%
#Para ajustar primero sacamos los NaNs porque sino explota.
# Crear máscaras para identificar NaNs
mask_r = np.isnan(r)
mask_m = np.isnan(m)

# Crear una máscara combinada que sea True donde haya NaNs en cualquiera de los dos arrays
combined_mask = mask_r | mask_m

# Filtrar los arrays para eliminar los NaNs
r_sin_nan = r[~combined_mask]
m_sin_nan = m[~combined_mask]
#Corte planetario en masas
mask_mplanetaria = m_sin_nan < 13
m_sin_nan	= m_sin_nan[mask_mplanetaria]
r_sin_nan	= r_sin_nan[mask_mplanetaria]

#Corte radial por presión electrónica
mask_radial = r_sin_nan <1
r_sin_nan_chico = r_sin_nan[mask_radial]
m_sin_nan_chico = m_sin_nan[mask_radial]

#	Ajustar los datos

lr = np.log10(r_sin_nan_chico)
lm = np.log10(m_sin_nan_chico)

ajuste = np.polyfit(lr, lm, 3, full=True, cov=True)
print("Coeficientes del ajuste:")
print(f"a: {ajuste[0][0]}")
print(f"b: {ajuste[0][1]}")
print(f"c: {ajuste[0][2]}")
print(f"d: {ajuste[0][3]}")

# Calcular el chi cuadrado
chi_squared = np.sum((lm - np.polyval(ajuste[0], lr))**2)
print(f"Chi cuadrado: {chi_squared}")

plt.plot(lr, lm, '+', markersize=2)
plt.plot(lr, np.polyval(ajuste[0], lr), '+')
plt.xlabel('Radio (R$_\mathrm{J}$)')
plt.ylabel('Masa (M$_\mathrm{J}$)')
plt.title('Masa vs Radio')
plt.show()


#Primero pero lineal:

ajuste = np.polyfit(lr, lm, 1, full=True, cov=True)
print("Coeficientes del ajuste:")
print(f"a: {ajuste[0][0]}")
print(f"b: {ajuste[0][1]}")

# Calcular el chi cuadrado
chi_squared = np.sum((lm - np.polyval(ajuste[0], lr))**2)
print(f"Chi cuadrado: {chi_squared}")

plt.plot(lr, lm, '+', markersize=2)
plt.plot(lr, np.polyval(ajuste[0], lr), '+')
plt.xlabel('Radio (R$_\mathrm{J}$)')
plt.ylabel('Masa (M$_\mathrm{J}$)')
plt.title('Masa vs Radio')
plt.show()

#Segundo ajuste

#Corte radial por presión electrónica para arriba
mask_radial2 = r_sin_nan > 1
r_sin_nan_grande = r_sin_nan[mask_radial2]
m_sin_nan_grande = m_sin_nan[mask_radial2]

#	Ajustar los datos

lr2 = np.log10(r_sin_nan_grande)
lm2 = np.log10(m_sin_nan_grande)

ajuste = np.polyfit(lr2, lm2, 0, full=True, cov=True)
print("Coeficientes del ajuste:")
print(f"a: {ajuste[0][0]}")
#print(f"b: {ajuste[0][1]}")

# Calcular el chi cuadrado
chi_squared = np.sum((lm2 - np.polyval(ajuste[0], lr2))**2)
print(f"Chi cuadrado: {chi_squared}")

plt.plot(lr2, lm2, '+', markersize=2)
plt.plot(lr2, np.polyval(ajuste[0], lr2), '+')
plt.xlabel('Radio (R$_\mathrm{J}$)')
plt.ylabel('Masa (M$_\mathrm{J}$)')
plt.title('Masa vs Radio')
plt.show()


# %% [markdown]
# Ejercicio 5

# %%
edad = datos['star_age']

plt.plot(m,	edad, 'o', markersize=2)
plt.xlabel('Masa (M$_\mathrm{J}$)')
plt.ylabel('Edad (Gy)')
plt.xscale('log')
#plt.yscale('log')
plt.title('Masa vs Edad')
plt.show()

# %% [markdown]
# Las más viejas parecen tener menos planetas.


