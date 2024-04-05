
import tkinter as tk
from tkinter import filedialog
import os

predefined_points = [4, [0,0], [3,0], [3,3], [0,3], 4, [0,0], [3,0], [1,1], [0,3], 0]

def is_clockwise(x1, y1, x2, y2, x3, y3):
    """
    Verifica se a curva definida por três pontos está virando para a direita.
    """
    value = (x2 - x1)*(y3 - y1) - (y2 - y1)*(x3 - x1)
    if value < 0:
        return True  # curva para a direita
    else:
        return False  # curva para a esquerda

def has_critical_point(points):
    """
    Verifica se um conjunto de pontos possui um ponto crítico.
    """
    primary_validation = True
    num_points = len(points)

    for i in range(num_points):
        j = 0
        x1 = points[i][0]
        y1 = points[i][1]
        if (num_points - i) > 1:
            x2 = points[i+1][0]
            y2 = points[i+1][1]
        else:
            x2 = points[abs(i-num_points)][0]
            y2 = points[abs(i-num_points)][1]
            j = 1
        if (num_points - i) > 2:
            x3 = points[i+2][0]
            y3 = points[i+2][1]
        elif j == 0:
            x3 = points[0][0]
            y3 = points[0][1]
        else:
            x3 = points[1][0]
            y3 = points[1][1]

        validation = is_clockwise(x1, y1, x2, y2, x3, y3)
        if i == 0:
            primary_validation = validation

        if primary_validation != validation:
            return True

    return False

def get_points():
    """
    Solicita pontos ao usuário.
    """
    points_list = []

    print(
"""
Preencha no seguinte formato:
4
0 0
3 0
3 3
0 3
...
Use '0' para finalizar
""")

    print("Digite os pontos:")
    while True:
        line = input()
        line = line.strip()
        if line == '0':
            points_list.append(int(line))
            break
        if ' ' in line:
            a = line.split()
            x = int(a[0])
            y = int(a[1])
            points_list.append([x, y])
        else:
            points_list.append(int(line))
    return points_list

def get_list_from_set(points, index):
    """
    Retorna uma galeria de pontos a partir de uma lista de pontos.
    """
    list_points = []
    index_aux = 0

    for j in range(len(points)):
        value = points[j]

        if isinstance(value, int):
            if value == 0:
                return 0
            if index_aux == index:
                for k in range(value):
                    list_points.append(points[j+k+1])
                return list_points
            index_aux += 1

    return list_points

def main():
    os.system('cls')
    """
    Função principal do programa.
    """
    print("Selecione uma opção:")
    print("1 - Input de pontos")
    print("2 - Conjunto de pontos predefinidos")

    opcao = int(input("Opção: "))
    os.system('cls')
    if opcao == 1:
        pontos = get_points()
    elif opcao == 2:

        for i, ponto in enumerate(predefined_points):
            if isinstance(ponto, int):
                print(ponto)
            else:
                print(f"{ponto[0]} {ponto[1]}")
        pontos = predefined_points
    else:
        print("Opção inválida.")
        return  # Sai da função se a opção for inválida

    index = 0

    print("\n")
    while True:
        galeriaPontos = get_list_from_set(pontos, index)
        if galeriaPontos == 0:
            break
        if has_critical_point(galeriaPontos):
            print("YES")
        else:
            print("NO")
        index += 1

if __name__ == "__main__":
    main()
