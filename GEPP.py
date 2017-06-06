# -*- coding: utf-8 -*-

import numpy as np

def AugmentedMatrix(A,b):
	# Cria uma nova matriz com uma coluna a mais
	Ab = np.zeros((A.shape[0], A.shape[1]+1))
	# Copia a matriz original para a nova matriz
	Ab[:,:-1] = A
	# Copia o vetor, convertido em uma matriz de uma coluna, para a nova matriz.
	Ab[:,-1:] = b.reshape(A.shape[0],1)

	return Ab

def GaussianEliminationPP(A,b):
# Soluciona um sistema Ax = b usando pivotação parcial e a matriz aumentada do sistema.
# Entrada: Matriz A de coeficientes, não singular, e b, vetor de escalares independentes.
# Saída: Vetor x de soluções.
	Ab = AugmentedMatrix(A,b)

	n = len(Ab)
	
	for i in range(0, n):
		# PIVOTAÇÃO
		# Procura pelo máximo na coluna atual
		pivot = abs(Ab[i,i])
		maxindex = i

		for k in range(i+1, n):
			if abs(Ab[k,i])>pivot:
				pivot = abs(Ab[k,i])
				maxindex = k

		# Faz a troca da linha pivô com a linha atual
		for k in range(i, n+1):
			swap = Ab[maxindex, k]
			Ab[maxindex, k] = Ab[i,k]
			Ab[i,k] = swap

		# ELIMINAÇÃO
		for k in range(i+1, n):
			M = Ab[k,i]/Ab[i][i]

			for j in range(i, n+1):
				if i==j:
					Ab[k,j]=0
				else:
					Ab[k,j] = Ab[k,j] - (M*Ab[i,j])

	# SUBSTITUIÇÃO REGRESSIVA (Solução de Ax = b)
	x = [0 for i in range(n)]
	for i in range(n-1, -1, -1):
		x[i] = Ab[i,n]/Ab[i,i]
		for k in range(i-1, -1, -1):
			Ab[k,n] -= Ab[k,i] * x[i]

	return x 

