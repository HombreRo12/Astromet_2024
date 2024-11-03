# %%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# %%
datos = pd.read_csv('data.csv')
magnitud = datos.iloc[:, 0]
funcionlum = datos.iloc[:, 1]
plt.scatter(magnitud, funcionlum, marker='+', color='red')
plt.yscale('log')

# %%
def	schechter(x,M,phi,alpha):
				m = 0.4*np.log(10)*phi*(10**(-0.4*(x-M)*(alpha+1)))*np.exp(-10**(-0.4*(x-M)))
				return m

# %%
xgraf = np.linspace(-23, -15, 100)
ygraf = schechter(xgraf, -20.83, 1.46*(10**-2), -1.2)

erromin = datos.iloc[:,2]
erromax = datos.iloc[:,3]


plt.plot(xgraf, ygraf, label='Función de Schechter', color='blue')
plt.errorbar(magnitud,funcionlum,yerr=[erromin, erromax],color='red', alpha=0.8, fmt='o', label='Datos', markersize=4)
plt.yscale('log')
plt.xlabel('Magnitud')
plt.ylabel('Funcion de luminosidad')
plt.title('Funcion de luminosidad vs Magnitud')
plt.legend()
plt.show()

# %% [markdown]
# Ahora, probamos variando los parámetros, uno a la vez, para ver como afectan al resultado:

# %%
xgraf = np.linspace(-23, -15, 100)
#Variamos M
for M in np.linspace(-25, -15, 10):
	ygraf = schechter(xgraf, M, 1.46*(10**-2), -1.2)
	plt.plot(xgraf, ygraf, label=f'M={M:.2f}')

erromin = datos.iloc[:, 2]
erromax = datos.iloc[:, 3]

plt.errorbar(magnitud, funcionlum, yerr=[erromin, erromax], color='red', alpha=0.8, fmt='o', label='Datos', markersize=4)
plt.yscale('log')
plt.xlabel('Magnitud')
plt.ylabel('Funcion de luminosidad')
plt.title('Funcion de luminosidad vs Magnitud')
plt.legend()
plt.show()


# %% [markdown]
# Me zarpé, reduzcamos:

# %%
xgraf = np.linspace(-23, -15, 100)
#Variamos M
for M in np.linspace(-21.5, -20, 10):
	ygraf = schechter(xgraf, M, 1.46*(10**-2), -1.2)
	plt.plot(xgraf, ygraf, label=f'$M_{{*}}={M:.2f}$')

erromin = datos.iloc[:, 2]
erromax = datos.iloc[:, 3]

plt.errorbar(magnitud, funcionlum, yerr=[erromin, erromax], color='red', alpha=0.8, fmt='o', label='Datos', markersize=4)
plt.yscale('log')
plt.xlabel('Magnitud')
plt.ylabel('Funcion de luminosidad')
plt.title('Funcion de luminosidad vs Magnitud')
plt.legend()
plt.show()

# %% [markdown]
# Es razonable, entonces, entre -21 y -20.6 irá M

# %% [markdown]
# Ahora, veamos $\phi$

# %%
xgraf = np.linspace(-23, -15, 100)
#Variamos phi
for phi in np.linspace(0.5*(10**-2), 2.5*(10**-2), 10):
	ygraf = schechter(xgraf, -20.83, phi, -1.2)
	plt.plot(xgraf, ygraf, label=f'$\phi_{{*}}$={phi:.4f}')

erromin = datos.iloc[:, 2]
erromax = datos.iloc[:, 3]

plt.errorbar(magnitud, funcionlum, yerr=[erromin, erromax], color='red', alpha=0.8, fmt='o', label='Datos', markersize=4)
plt.yscale('log')
plt.xlabel('Magnitud')
plt.ylabel('Funcion de luminosidad')
plt.title('Funcion de luminosidad vs Magnitud')
plt.legend()
plt.show()

# %% [markdown]
# Razonable, por lo tanto $\phi \in [1.2,1.6]10^{-2}$

# %% [markdown]
# Finalmente, veamos $\alpha$

# %%
xgraf = np.linspace(-23, -15, 100)
# Variamos alpha
for alpha in np.linspace(-1.5, -0.5, 10):
	ygraf = schechter(xgraf, -20.83, 1.46*(10**-2), alpha)
	plt.plot(xgraf, ygraf, label=f'$\\alpha={alpha:.4f}$')

erromin = datos.iloc[:, 2]
erromax = datos.iloc[:, 3]

plt.errorbar(magnitud, funcionlum, yerr=[erromin, erromax], color='red', alpha=0.8, fmt='o', label='Datos', markersize=4)
plt.yscale('log')
plt.xlabel('Magnitud')
plt.ylabel('Función de luminosidad')
plt.title('Función de luminosidad vs Magnitud')
plt.legend()
plt.show()

# %% [markdown]
# Definamos entonces lo necesario para el algoritmo:

# %%
def loglike(x, y, M, phi, alpha, sigma):
				m = schechter(x, M, phi, alpha)
				L = -(1/2) * np.sum(((y - m) / sigma) ** 2)
				return L

# %%
fiducial_M = -20.83
fiducial_phi = 1.46e-2
fiducial_alpha = -1.2

alphamin = fiducial_alpha*(1+0.1)
alphamax = fiducial_alpha*(1-0.1)
Mmin = fiducial_M*(1+0.1)
Mmax = fiducial_M*(1-0.1)
phimin = fiducial_phi*(1-0.2)
phimax = fiducial_phi*(1+0.2)
print(alphamin, alphamax, Mmin, Mmax, phimin, phimax)

# %%
bordeinferr =	np.array([Mmin, phimin, alphamin])
bordesuperr = np.array([Mmax, phimax, alphamax])

def priors(params):
	if np.all(bordeinferr <= params) and np.all(params <= bordesuperr):
		return 0
	else:
		return -np.inf

# %%
def post(x, y, M, phi, alpha, sigma):
	params = np.array([M, phi, alpha])
	prior_prob = priors(params)
	return loglike(x, y, M, phi, alpha, sigma) + prior_prob

# %% [markdown]
# Normalizemos

# %%
def normM(M):
    return (M - Mmin) / ( Mmax - Mmin)

def anti_normM(M):
				return (M * (Mmax - Mmin)) + Mmin

def normphi(phi):
				return (phi - phimin) / (phimax - phimin)

def anti_normphi(phi):
				return (phi * (phimax - phimin)) + phimin

def normalpha(alpha):
				return (alpha - alphamin) / (alphamax - alphamin)

def anti_normalpha(alpha):
				return (alpha * (alphamax - alphamin)) + alphamin

# %% [markdown]
# Ahora, armemos el algoritmo:

# %%
def mc(x, y, sigma, N):
	
	#Lanzamos en un lugar al azar dentro de los priors.
	M = np.random.uniform(Mmin, Mmax)
	phi = np.random.uniform(phimin, phimax)
	alpha = np.random.uniform(alphamin, alphamax)
	
	print('Initial values: ', M, phi, alpha)

	#Armamos las listas donde guardaremos todo
	Mreccorridos = np.zeros(N+1)
	phirecorridos = np.zeros(N+1)
	alpharecorridos = np.zeros(N+1)
	p = np.zeros(N+1)
	
	#Calculamos el primer valor de la probabilidad
	p[0] = post(x, y, M, phi, alpha, sigma)
	print(p[0])
	
	#Guardamos los primeros valores
	Mreccorridos[0] = M
	phirecorridos[0] = phi
	alpharecorridos[0] = alpha
	
	#Empezamos a recorrer
	for i in range(1,N+1):

		#Damos pasos inversamente proporcionales a la probabilidad, de modo que sean largos lejos del pico, y cortos cerca del pico.
		if i <	1000:
			pp	= 1e-1
		elif i < 3000:
			pp = 1e-2
		else:
			pp = 1e-3

		M_paso = normM(M) + np.random.uniform(-1,1)*pp
		phi_paso = normphi(phi) + np.random.uniform(-1,1)*pp
		alpha_paso = normalpha(alpha) + np.random.uniform(-1,1)*pp
		
		#Ahora, desonormalizamos para calcular la probabilidad:
		M_paso = anti_normM(M_paso)
		phi_paso = anti_normphi(phi_paso)
		alpha_paso = anti_normalpha(alpha_paso)

		#Ahora, si la probabilidad del paso es mayor a la probabilidad actual, nos movemos a ese paso.
		current_post = post(x, y, M_paso, phi_paso, alpha_paso, sigma)
		dif = current_post - p[i-1]
		print(dif)
		ran = np.log(np.random.uniform(0,1))
		if  dif > ran:
			M = M_paso
			phi = phi_paso
			alpha = alpha_paso
		
		Mreccorridos[i] = M
		phirecorridos[i] = phi
		alpharecorridos[i] = alpha
		p[i] = post(x, y, M, phi, alpha, sigma)

	#Obtengo todos los valores de los parametros
	return	Mreccorridos, phirecorridos, alpharecorridos, p

# %%
sigma = (erromax-erromin)/2
mcad, phicad, alphacad, logpcad = mc(magnitud, funcionlum, sigma, 10000)

# %%
f, axs = plt.subplots(1, 3, figsize=(25, 7))

# Gráfico de M vs phi
sc0 = axs[0].scatter(mcad[-2000:], phicad[-2000:], marker='.', s=10, c=range(2000), cmap='brg')
cbar0 = plt.colorbar(sc0, ax=axs[0])
cbar0.set_label('Paso')
axs[0].set_xlabel('M')
axs[0].set_ylabel('$\phi$')
axs[0].set_title('Espacio de parámetros M vs $\phi$')

# Gráfico de M vs alpha
sc1 = axs[1].scatter(mcad[-2000:], alphacad[-2000:], marker='.', s=10, c=range(2000), cmap='brg')
cbar1 = plt.colorbar(sc1, ax=axs[1])
cbar1.set_label('Paso')
axs[1].set_xlabel('M')
axs[1].set_ylabel('$\\alpha$')
axs[1].set_title('Espacio de parámetros M vs $\\alpha$')

# Gráfico de phi vs alpha
sc2 = axs[2].scatter(phicad[-2000:], alphacad[-2000:], marker='.', s=10, c=range(2000), cmap='brg')
cbar2 = plt.colorbar(sc2, ax=axs[2])
cbar2.set_label('Paso')
axs[2].set_xlabel('$\phi$')
axs[2].set_ylabel('$\\alpha$')
axs[2].set_title('Espacio de parámetros $\phi$ vs $\\alpha$')

plt.show()

# %%
f, axs = plt.subplots(1, 3, figsize=(20, 7))

#Valor por cada paso
axs[0].plot(mcad)
axs[0].set_xlabel('Paso')
axs[0].set_ylabel('M')
axs[0].set_title('Valor de M por paso')

axs[1].plot(phicad)
axs[1].set_xlabel('Paso')
axs[1].set_ylabel('$\phi$')
axs[1].set_title('Valor de $\phi$ por paso')

axs[2].plot(alphacad)
axs[2].set_xlabel('Paso')
axs[2].set_ylabel('$\\alpha$')
axs[2].set_title('Valor de $\\alpha$ por paso')


# %%
mbueno = np.sum(mcad[-2000:])/2000
phibueno = np.sum(phicad[-2000:])/2000
alphabueno = np.sum(alphacad[-2000:])/2000
print(mbueno, phibueno, alphabueno)

# %%
mix = 20
N = 10000
mcaden, phicaden, alphacaden, logpcaden = np.zeros((N+1,	mix)), np.zeros((N+1, mix)), np.zeros((N+1, mix)), np.zeros((N+1, mix))

for j in range(mix):
    mcaden[:,j], phicaden[:,j], alphacaden[:,j], logpcaden[:,j] = mc(magnitud, funcionlum, sigma, N)

# %%
colores = {0:'red', 1:'blue', 2:'green', 3:'yellow', 4:'purple', 5:'orange', 6:'black', 7:'pink', 8:'brown', 9:'cyan', 10:'magenta', 11:'gray', 12:'olive', 13:'lime', 14:'teal', 15:'navy', 16:'maroon', 17:'aqua', 18:'fuchsia', 19:'silver'}

f, axs = plt.subplots(1, 3, figsize=(20, 7))

# Gráfico de M vs likelihood
for j in range(mix):
				axs[0].scatter(mcaden[-2000:,j],	logpcaden[-2000:,j], marker='.', s=10, color=colores[j],label=f'Cadena {j+1}', alpha=0.8)
axs[0].set_xlabel('M')
axs[0].set_ylabel('log(P)')
axs[0].set_title('M vs log Likelihood')
axs[0].legend()
# Gráfico de alpha vs likelihood
for j in range(mix):
				axs[1].scatter(alphacaden[-2000:,j], logpcaden[-2000:,j], marker='.', s=10, color=colores[j],label=f'Cadena {j+1}', alpha=0.8)
axs[1].set_ylabel('log(P)')
axs[1].set_xlabel('$\\alpha$')
axs[1].set_title('$\\alpha$ vs log Likelihood')
axs[1].legend()
# Gráfico de phi vs likelihood

for j in range(mix):
				axs[2].scatter(phicaden[-2000:,j], logpcaden[-2000:,j], marker='.', s=10, color=colores[j],label=f'Cadena {j+1}', alpha=0.8)
axs[2].set_ylabel('log(P)')
axs[2].set_xlabel('$\phi$')
axs[2].set_title('$\phi$ vs log Likelihood')
axs[2].legend()
plt.show()

# %%
f, axs = plt.subplots(1, 3, figsize=(20, 7))

# Gráfico de M vs paso
for j in range(mix):
	axs[0].plot(mcaden[:, j], marker='.', markersize=2, color=colores[j], label=f'Cadena {j+1}', alpha=0.8)
axs[0].set_xlabel('Paso')
axs[0].set_ylabel('M')
axs[0].set_title('M vs Eslabón')
axs[0].legend(loc='center right')

# Gráfico de alpha vs eslabón
for j in range(mix):
	axs[1].plot(alphacaden[:, j], marker='.', markersize=2, color=colores[j], label=f'Cadena {j+1}', alpha=0.8)
axs[1].set_xlabel('Paso')
axs[1].set_ylabel('$\\alpha$')
axs[1].set_title('$\\alpha$ vs Eslabón')
axs[1].legend(loc='center right')

# Gráfico de phi vs eslabón
for j in range(mix):
	axs[2].plot(phicaden[:, j], marker='.', markersize=2, color=colores[j], label=f'Cadena {j+1}', alpha=0.8)
axs[2].set_xlabel('Paso')
axs[2].set_ylabel('$\phi$')
axs[2].set_title('$\phi$ vs Eslabón')
axs[2].legend(loc='center right')

plt.show()

# %%
f, axs = plt.subplots(1, 3, figsize=(20, 7))

# Histograma de M
for j in range(mix):
	axs[0].hist(mcaden[-5000:, j], color=colores[j], label=f'Cadena {j+1}', alpha=0.8)
axs[0].set_xlabel('M')
axs[0].set_ylabel('Frecuencia')
axs[0].set_title('Marginalización de M')
axs[0].legend(loc='center right')

# Histograma de alpha
for j in range(mix):
	axs[1].hist(alphacaden[-5000:, j], color=colores[j], label=f'Cadena {j+1}', alpha=0.8)
axs[1].set_xlabel('$\\alpha$')
axs[1].set_ylabel('Frecuencia')
axs[1].set_title('Marginalización de $\\alpha$')
axs[1].legend(loc='center right')

# Histograma de phi
for j in range(mix):
	axs[2].hist(phicaden[-5000:, j], color=colores[j], label=f'Cadena {j+1}', alpha=0.8)
axs[2].set_xlabel('$\phi$')
axs[2].set_ylabel('Frecuencia')
axs[2].set_title('Marginalización de $\phi$')
axs[2].legend(loc='center right')



plt.show()

# %%
distancias = np.zeros((N+1, mix, mix))
#Calcular todas las distancias en cada paso
for i in range(N+1):
    for j in range(mix):
        for k in range(j+1, mix):
            distancias[i, j, k] = np.linalg.norm([mcaden[i, j] - mcaden[i, k], phicaden[i, j] - phicaden[i, k], alphacaden[i, j] - alphacaden[i, k]])

ejex = np.linspace(0, N, N+1)
# Calcular la distancia máxima en cada paso
ejey = np.max(distancias, axis=(1, 2))

# Graficar
plt.plot(ejex, ejey, label='Máxima Distancia entre Cadenas')
plt.xlabel('Paso')
plt.ylabel('Distancia')
plt.legend()
plt.title('Distancia entre Cadenas en función del Paso')
plt.show()

print('Máxima distancia entre cadenas:', np.max(ejey))

# %%
import pandas as pd

mmedio = np.zeros(mix)
phimedio = np.zeros(mix)
alphamedio = np.zeros(mix)
mvari	= np.zeros(mix)
phivari = np.zeros(mix)
alphavari = np.zeros(mix)


for i in range(mix):
    mmedio[i] = np.mean(mcaden[-6000:, i])
    phimedio[i] = np.mean(phicaden[-6000:, i])
    alphamedio[i] = np.mean(alphacaden[-6000:, i])
    mvari[i]	= np.std(mcaden[-6000:, i])
    phivari[i] = np.std(phicaden[-6000:, i])
    alphavari[i] = np.std(alphacaden[-6000:, i])

mgod = np.mean(mmedio)
phigod = np.mean(phimedio)
alphagod = np.mean(alphamedio)
mvarigod	= np.mean(mvari)
phivarigod = np.mean(phivari)
alphavarigod = np.mean(alphavari)
errorm = mvarigod/np.sqrt(mix)
errorphi = phivarigod/np.sqrt(mix)
erroralpha = alphavarigod/np.sqrt(mix)


print(mgod, phigod, alphagod)
print(errorm, errorphi, erroralpha)
# Crear un DataFrame con los valores medios
df_medios = pd.DataFrame({
	'M medio': mmedio,
	'Phi medio': phimedio,
	'Alpha medio': alphamedio
})

# Convertir  a una tabla de LaTeX para el informe
latex_table = df_medios.to_latex(index=False, float_format="%.6f")
print(latex_table)


