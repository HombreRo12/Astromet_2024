# %% [markdown]
# a) Todas las agujas se lanzan con una distribución uniforme, por lo que las distribuciones para $(x,\theta)$ deben ser constantes. Calculemos para $x$: Dado que se mide desde el centro de la aguja y la mesa mide t, la máxima distancia posible es $\frac{t}{2} \implies \int_{0}^{\frac{t}{2}}p(x)dx = 1 \leftrightarrow \frac{t}{2} \alpha = 1 \leftrightarrow \alpha = \frac{2}{t} \implies p(x)=\frac{2}{t}$
# 
# Análogamente, $\theta \in [0,\frac{\pi}{2}] \implies p(\theta)=\frac{2}{\pi}$
# 
# $\therefore p(x,\theta)=\frac{4}{t\pi}$ pues son independientes.

# %% [markdown]
# b) Si $l<t$ la aguja tocará la raya cuando $cos(\frac{\pi}{2}-\theta)=\frac{x}{\frac{l}{2}}$ (trigonometría) por lo tanto, para que se produzca un contacto necesitamos que $\frac{l}{2}sen(\theta) \geq x \implies $ debemos integrar en esa región para x:
# 
# $\int_{0}^{\frac{\pi}{2}}\int_{0}^{\frac{l}{2}sen(\theta)}p(x,\theta)dxd\theta = \int_{0}^{{\frac{\pi}{2}}} \frac{l}{2}sen(\theta)\frac{2}{t} =\frac{2l}{t\pi}$ es la probabilidad de que toque una raya.
# 

# %% [markdown]
# c) $p=\frac{2l}{t\pi}\leftrightarrow \pi = \frac{2l}{tp}$ por lo tanto basta simular el experimento para poder estimar $\pi$.

# %%
import numpy as np

t=2 #Distancia entre rayas.
l=1 #Longitud de la aguja, respetando que l<t.
lanzamientos = 10000000 #Cantidad de lanzamientos.
contador = 0 #Contador de contactos.

for i in range(lanzamientos):
    x=np.random.uniform(0,t/2) #Genero un número aleatorio entre los permitidos para x.
    a=np.random.uniform(0,np.pi/2) #Genero un número aleatorio entre los permitidos para theta.
    if x<=l/2*np.sin(a): #Separo los casos donde se toca la aguja con la raya.
        contador+=1

#Calculo el valor de pi.
pi=2*l*lanzamientos/(t*contador)

print(pi)

# %% [markdown]
# Obteniendo un resultado cercano a pi.


