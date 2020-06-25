"""
Tarea 3 Modelos Probibalísticos de Señales y Sistemas
Jeremy Alvarado D
B40252
"""

import pandas
import pandas as pd
import numpy as np 
from scipy import stats 
import matplotlib.pyplot as plt
import scipy.stats as stats
from fitter import Fitter
import math
from scipy.stats.contingency import margins
from scipy.stats import norm
from scipy.optimize import curve_fit
from scipy.stats import norm

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D


#Importo el archivo lote.csv 
mydataset1 = pd.read_csv('xy.csv', names=[ '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25'], header = 0 )
mydataset2 = pd.read_csv('xyp.csv', names=['x', 'y', 'p'], header = 0 )


#1) Modelo ajustado (Gaussiano))
fX = np.sum(mydataset1, axis=0)
fY = np.sum(mydataset1, axis=1)



#print('fY \n', np.sum(mydataset1, axis=0))
#print('fX \n', np.sum(mydataset1, axis=1))

#Gráficas de los datos

plt.plot(fX)
plt.xlabel('Valores X')
plt.ylabel('Valores Y')
plt.title('Curva de los datos fY')
plt.show()


plt.plot(fY)
plt.xlabel('Valores X')
plt.ylabel('Valores Y')
plt.title('Curva de los datos fX')
plt.show()



#Obtener los parámetros µ y σ para de forma manual para obtener el modelo del
# mejor ajuste que este caso es Gaussiano 

print('')

print('Parámetros de fX')

medx = sum(mydataset2.x * mydataset2.p) / sum(mydataset2.p)

print('µ =', medx)

sigx = np.sqrt( (sum(mydataset2.x**2 * mydataset2.p) /  sum(mydataset2.p) ) - medx**2 )
print('σ =', sigx)


print('')

print('Parámetros de fY')

medy = sum(mydataset2.y * mydataset2.p)

print('µ ', medy)

sigy = np.sqrt( (sum(mydataset2.y**2 * mydataset2.p) /  sum(mydataset2.p) ) - medy**2 )
print('σ =', sigy)


#2) Función de densidad conjunta
#Es la multiplicación de fX*fY pero como son de diferente tamaño lo mejor será 
#tomar los modelos y multiplicarlos.
#Es decir gauss1 * gauss2 

#3) Correlación, Covarianza, Pearson

correlacion = sum(mydataset2.x * mydataset2.y * mydataset2.p)

print('')
print('Correlaión =', correlacion)

covarianza = ( correlacion / sum(mydataset2.p) ) - (medx * medy)

print('')
print('Covarianza =', covarianza)

pearson = (covarianza)/(sigx * sigy)

print('')
print('Coeficiente de Pearson =', pearson)



#Gráficas de los modelos de mejor ajuste

x_axis = np.arange(5, 15, 0.001)

plt.plot(x_axis, norm.pdf(x_axis,medx,sigx))
plt.xlabel('Valores X')
plt.ylabel('Valores Y')
plt.title('Modelo del mejor ajuste fX')
plt.show()


x_axis = np.arange(5, 25, 0.001)
plt.plot(x_axis, norm.pdf(x_axis,medy,sigy))
plt.xlabel('Valores X')
plt.ylabel('Valores Y')
plt.title('Modelo del mejor ajuste fY')
plt.show()

#4) Figura en 3D 


f = plt.figure()
ax = plt.axes(projection= '3d')
X, Y = np.meshgrid(range(5,16), range(5,26)) 
fdc = norm.pdf(X,medx,sigx) * norm.pdf(Y,medy,sigy)
ax.plot_surface(X,Y, fdc, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')
ax.set_title('Función de densidad conjunta 3D');
plt.show()


