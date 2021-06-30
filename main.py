import numpy as np 

print("Primer Ejercicio")
X = np.random.rand(2, 2)
Y = X - X.mean(axis=1, keepdims=True)
Y = X - X.mean(axis=1).reshape(-1, 1)
print (Y)
print("****************************************************")

print("Segundo ejercicio")
A = np.random.randint(0,10,(3,4,3,4))
sum = A.reshape(A.shape[:-2] + (-1,)).sum(axis=-1)
print(sum)
print("****************************************************")

print("Tercer Ejercicio")
Z = np.ones(10)
I = np.random.randint(0,len(Z),20)
Z += np.bincount(I, minlength=len(Z))
print(Z)
print("****************************************************")
print("ejercicio 4")
for dtype in [np.int8, np.int32, np.int64]:
  print(np.iinfo(dtype).min)
  print(np.iinfo(dtype).max)
for dtype in [np.float32, np.float64]:
  print(np.finfo(dtype).min)
  print(np.finfo(dtype).max)
  print(np.finfo(dtype).eps)
  print("****************************************************")
print("ejercicio 5")
Z = np.arange(100)
v = np.random.uniform(0,100)
index = (np.abs(Z-v)).argmin()
print(Z[index])
print("****************************************************")
print("ejercicio 6")
Z = np.arange(9).reshape(3,3)
for index, value in np.ndenumerate(Z):
  print(index, value)
for index in np.ndindex(Z.shape):
  print(index, Z[index])
print("****************************************************")
print("Ejercicio 7")
print("Considerando el vector [1, 2, 3, 4, 5], como crearias un nuevo vector con 3 ceros consecutivos intercalados entre cada valor del vector dicho?")
arrayP = np.array([1, 2, 3, 4, 5]) #principal arreglo
nZeros = 3 #ceros a intercalar
arrayR = np.zeros(len(arrayP) + (len(arrayP-1)*nZeros)) #Se llena de ceros el numero de elementos del array resultante
#print(arrayR)
arrayR[::nZeros+1] = arrayP
print(arrayR)
print("****************************************************")
print("\nEjercicio 8")
print("Cómo intercambiar dos filas de un arreglo?")
B = np.arange(9).reshape(3,3) #arreglo con valores del 0 al 8 divididos en 3 filas y 3 columnas
print("Arreglo original")
print(B)
print("Arreglo cambiado")
B[[0, 1]] = B[[1, 0]]
print(B)
print("****************************************************")
print("\nEjercicio 9")
print("Considerando dos conjuntos de puntos C1, C2 que describen líneas tipo (2D) y un punto P, Cómo calcula la distancia de P a cada linea (C1[i], C2[i])?")
def distance(C1, C2, P):

    T = C2 - C1
    L = (T**2).sum(axis=1)
    U = -((C1[:,0]-P[...,0])*T[:,0] + (C1[:,1]-P[...,1])*T[:,1]) / L
    U = U.reshape(len(U),1)
    D = C1 + U*T - P
    return np.sqrt((D**2).sum(axis=1))
C1 = np.random.uniform(-10,10,(10,2))
C2 = np.random.uniform(-10,10,(10,2))
print("Conjunto de puntos 1")
print(C1)
print("Conjunto de puntos 2")
print(C2)
P = np.random.uniform(-10,10,( 1,2))
print("Punto P")
print(P)
print("Distancia calculada entre los puntos")
print(distance(C1, C2, P))
print("****************************************************")
print("\nEjercicio 10")
print("Como obtener los valores 'n' valores mas largos de una matriz\n")
Z = np.arange(10000)
np.random.shuffle(Z)
n = 9
# Slow
print (Z[np.argsort(Z)[-n:]])
# Fast
print (Z[np.argpartition(-Z,n)[:n]])
print("\n")
print("****************************************************")
print("\nEjercicio 11")
print("Consideramos dos Arreglos A Y B de forma (8,3) y (2,2)Como encontrar filas de A que contengas elementos de cada fila de B independientemente del orden de los elementos en  B\n")
A = np.random.randint(0,5,(8,3))
B = np.random.randint(0,5,(2,2))
C = (A[..., np.newaxis, np.newaxis] == B)
rows = (C.sum(axis=(1,2,3)) >= B.shape[1]).nonzero()[0]
print(rows)
print("\n")
print("****************************************************")
print("\nEjercicio 12")
print("Dada un amatriz bidimencional, como extraer filas unicas?\n")

Z = np.random.randint(0,2,(6,3))
T = np.ascontiguousarray(Z).view(np.dtype((np.void, Z.dtype.itemsize * Z.shape[1])))
_, idx = np.unique(T, return_index=True)
uZ = Z[idx]
print(uZ)

# 74 como encontrar el valor con mas frecuencia en el array
print("¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨13¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨")
Z = np.random.randint(0,10,50)
print(Z)
print("Valor con el valor con mas freq: ",np.bincount(Z).argmax())
print("Imprimo el valor que menos se freq: ",np.bincount(Z).argmin())


# 77 Considere un conjunto de p matrices con forma (n, n) y un conjunto de p vectores con forma
# (n, 1). ¿Cómo calcular la suma de los p productos de la matriz a la vez? (el resultado tiene
# forma
print("¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨14¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨")
p, n = 10, 20
M = np.ones((p,n,n))
V = np.ones((p,n,1))
S = np.tensordot(M, V, axes=[[0, 2], [0, 1]])#tensordot?
print(S)


# 78 Considere una matriz de 16x16, ¿cómo obtener la suma de bloques (el tamaño del bloque es 4x4)?
print("¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨15¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨")
Z = np.ones((16,16))
print(Z)
k = 4
S = np.add.reduceat(np.add.reduceat(Z, np.arange(0, Z.shape[0], k),
axis=0),np.arange(0, Z.shape[1], k), axis=1)#para reducir esta matriz a 4
print(S)


# 79 How to implement the Game of Life using numpy arrays?
print("¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨16¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨¨")
def iterate(Z):
    N = (Z[0:-2,0:-2] + Z[0:-2,1:-1] + Z[0:-2,2:] +
         Z[1:-1,0:-2] + Z[1:-1,2:] +
         Z[2: ,0:-2] + Z[2: ,1:-1] + Z[2: ,2:])
# Apply rules
    birth = (N==3) & (Z[1:-1,1:-1]==0)
    survive = ((N==2) | (N==3)) & (Z[1:-1,1:-1]==1)
    Z[...] = 0
    Z[1:-1,1:-1][birth | survive] = 1
    return Z
Z = np.random.randint(0,2,(50,50))
for i in range(100): Z = iterate(Z)

print(Z)


