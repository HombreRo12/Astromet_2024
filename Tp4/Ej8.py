# %%
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp

# %% [markdown]
# Primero, agarro la inversa de distrib exponencial que tenía de la guía 2 y simulo las 50 mediciones

# %%
def pdf(x,l):
    pdf = np.exp(-x*l)*l
    return pdf

def cdf(x,l):
    cdf = 1-np.exp(-x*l)
    return cdf

def inv(y,l):
    inv = -np.log(1-y)/l
    return inv

y=np.random.rand(50)
x=inv(y,5)

plt.hist(x,bins=50,density=True)
plt.xlabel('x')
plt.ylabel('Frecuencia')
plt.title('Distribución exponencial')
plt.show()

# %% [markdown]
# $L =\prod_{i=1}^N \lambda e^{-\lambda x_i} $ 
# 
# $\log{L}=N\log{\lambda}-\lambda \sum_{i=1}^N x_i$
# 
# $\lambda=\frac{1}{\overline{x}}$

# %%
maxlikelihood = 1/np.mean(x)

print('El valor de máximo likelihood para estimar lambda es:',maxlikelihood)

# %%
bordeinferr	= 3
bordesuperr	= 7

lambda_estimador = np.linspace(1, 10, 100)

def loglikelihood(x, l):
    L = np.sum(np.log(pdf(x, l)))
    return L

def priors(p):
	if bordeinferr <= p and p <= bordesuperr:
		return 0
	else:
		return -np.inf

def post(x, l):
    prior_prob = priors(l)
    return loglikelihood(x, l) + prior_prob

posterior	= np.zeros(len(lambda_estimador))
for i in range(len(lambda_estimador)):
				posterior[i] = post(x, lambda_estimador[i])

lambda_bayesiano =	lambda_estimador[np.argmax(posterior)]

print(posterior)
print('El valor de lambda que maximiza el posterior es:',lambda_bayesiano)


# %%
delta_frec = abs(5-maxlikelihood)
delta_bayes =	abs(5-lambda_bayesiano)

print(delta_bayes)
print(delta_frec)

porcen_frec	= (delta_frec/5)*100
porcen_bayes = (delta_bayes/5)*100

print('El error porcentual para el estimador frecuentista es:',porcen_frec)
print('El error porcentual para el estimador bayesiano es:',porcen_bayes)


# %% [markdown]
# Fluctua mucho, hagamos varias tiradas.

# %%
experimento = 10000

maxlikelihood_lista = np.zeros(experimento)
lambda_bayesiano_lista = np.zeros(experimento)

for j in range(experimento):

	y=np.random.rand(50)
	x=inv(y,5)

	maxlikelihood = 1/np.mean(x)
	maxlikelihood_lista[j] = maxlikelihood

	for i in range(len(lambda_estimador)):
					posterior[i] = post(x, lambda_estimador[i])

	lambda_bayesiano =	lambda_estimador[np.argmax(posterior)]
	lambda_bayesiano_lista[j] = lambda_bayesiano

frec_medio	= np.mean(maxlikelihood_lista)
bayes_medio	= np.mean(lambda_bayesiano_lista)

print('El valor medio del estimador frecuentista es:',frec_medio)
print('El valor medio del estimador bayesiano es:',bayes_medio)

delta_frec_lista	= abs(5-frec_medio)
delta_bayes_lista = abs(5-bayes_medio)
porcen_frec_lista	= (delta_frec_lista/5)*100
porcen_bayes_lista = (delta_bayes_lista/5)*100

print('Lambda frecuentista medio',frec_medio)
print('Lambda bayesiano medio',bayes_medio)
print('Error porcentual medio frecuentista',porcen_frec_lista)
print('Error porcentual medio bayesiano',porcen_bayes_lista)


