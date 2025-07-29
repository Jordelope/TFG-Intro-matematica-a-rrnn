"""
usar_red.py
------------

Este script permite cargar una red neuronal multicapa (MLP) previamente entrenada y guardada en disco, y evaluarla sobre un conjunto de test.
En este caso, la red se utiliza para predecir la posición de jugadores de la NBA a partir de sus estadísticas individuales.

Estructura general del script:
- Carga la red MLP desde un archivo JSON usando la función cargar_red.
- Carga los datos de test y las etiquetas reales desde procesar_datos_entrenamiento.py.
- Evalúa la red sobre el conjunto de test y decodifica las predicciones a etiquetas de posición (PG, SG, SF, PF, C).
- Calcula y muestra la precisión de la red, así como los errores si se desea.

Parámetros principales:
- nombre_archivo_red: ruta del archivo JSON con la red entrenada.
- mostrar_fallos: si es True, muestra los errores de predicción detallados.

El script está pensado para ser ejecutado como programa principal (__main__), mostrando los resultados de la evaluación por consola.
"""
from MLP import MLP, guardar_red, cargar_red
from proc_datos_entrenamiento import  Xs_test_def , Ys_test_def, nombre_set_test


## Funciones relevantes ##

def decodificar_pos(xs):
    """
    Decodifica la posición predicha a partir de un vector de salida one-hot.
    Devuelve el nombre de la posición correspondiente (PG, SG, SF, PF, C).
    Si la entrada es None, devuelve 'UNKNOWN'.
    """
    posiciones = ["PG", "SG", "SF", "PF", "C"]
    if xs is None:
        return "UNKNOWN"
    return posiciones[xs.argmax().item()]


#Establecemos el test: 
test = Xs_test_def
test_result = Ys_test_def
test_result_pos = [decodificar_pos(y) for y in test_result] # Las posiciones reales
set_a_usar = nombre_set_test


#####  Inicializamos red  #####
nombre_archivo_red = r"redes_disponibles\red_prueba_2capasGrandes.json"

mostrar_fallos = False


#------------------------------------------------------------------------------------------------------------------------------------------



if __name__ == "__main__":
    # Cargamos la red previamente entrenada
    NN = cargar_red(nombre_archivo_red)

    # Realizamos las predicciones sobre el conjunto de test
    pred_test = [NN(x) for x in test]
    pred_class_test = [decodificar_pos(x) for x in pred_test]
    test_result_pos = [decodificar_pos(y) for y in test_result]

    print(f"\nLa red {nombre_archivo_red} sobre el test {set_a_usar} dice: \n{pred_class_test}")
    print(f"Deberia dar: \n{test_result_pos}\n")

    # Cálculo de precisión y errores
    cont_buenos = 0
    errores = []
    for i in range(len(pred_class_test)):
        if pred_class_test[i] == test_result_pos[i]:
            cont_buenos += 1
        else:
            errores.append(i)
    accuracy = cont_buenos / len(pred_class_test)

    print(f"La red {nombre_archivo_red} ha acertado {cont_buenos} / {len(pred_class_test)}.")
    print(f"Precisión en test: {accuracy*100:.2f}%\n")

    # Mostrar detalles de los fallos si se desea
    if mostrar_fallos:
        for i in errores:
            fallo = pred_class_test[i]
            correcto = test_result_pos[i]
            print(f"La red ha fallado en el jugador {i}. Ha dicho que es un '{fallo}', pero en verdad es un '{correcto}.'")
