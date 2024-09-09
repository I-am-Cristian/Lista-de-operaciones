import numpy as np

# 1. Adición de vectores complejos
def suma_vectores_complejos(v1, v2):
    return np.add(v1, v2)

# 2. Inverso (aditivo) de un vector complejo
def inverso_aditivo_vector(v):
    return np.negative(v)

# 3. Multiplicación de un escalar por un vector complejo
def multiplicacion_escalar_vector(escalar, v):
    return np.dot(escalar, v)

# 4. Adición de matrices complejas
def suma_matrices_complejas(m1, m2):
    return np.add(m1, m2)

# 5. Inversa (aditiva) de una matriz compleja
def inverso_aditivo_matriz(m):
    return np.negative(m)

# 6. Multiplicación de un escalar por una matriz compleja
def multiplicacion_escalar_matriz(escalar, m):
    return np.dot(escalar, m)

# 7. Transpuesta de una matriz o vector
def transpuesta(m):
    return np.transpose(m)

# 8. Conjugada de una matriz o vector
def conjugada(m):
    return np.conjugate(m)

# 9. Adjunta (daga) de una matriz o vector
def adjunta(m):
    return np.transpose(np.conjugate(m))

# 10. Producto de dos matrices (de tamaños compatibles)
def producto_matrices(m1, m2):
    return np.matmul(m1, m2)

# 11. Acción de una matriz sobre un vector
def accion_matriz_sobre_vector(m, v):
    return np.dot(m, v)

# 12. Producto interno de dos vectores
def producto_interno(v1, v2):
    return np.vdot(v1, v2)

# 13. Norma de un vector
def norma_vector(v):
    return np.linalg.norm(v)

# 14. Distancia entre dos vectores
def distancia_entre_vectores(v1, v2):
    return np.linalg.norm(np.subtract(v1, v2))

# 15. Valores y vectores propios de una matriz
def valores_vectores_propios(m):
    valores_propios, vectores_propios = np.linalg.eig(m)
    return valores_propios, vectores_propios

# 16. Revisar si una matriz es unitaria
def es_unitaria(m):
    identidad = np.eye(m.shape[0])
    return np.allclose(np.matmul(adjunta(m), m), identidad)

# 17. Revisar si una matriz es Hermitiana
def es_hermitiana(m):
    return np.allclose(m, adjunta(m))

# 18. Producto tensorial de dos matrices o vectores
def producto_tensor(m1, m2):
    return np.kron(m1, m2)
