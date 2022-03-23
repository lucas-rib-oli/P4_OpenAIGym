# OpenAI Gym
Tercer entregable para la asignatura de Simuladores de Robots del Máster de Robótica y Automatización (UC3M)

## Extras
- Extra 1: Se ha creado el siguiente repositorio en [GitHub](https://github.com/lucas-rib-oli/P4_OpenAIGym) para almacenar los cambios en el código. 
- Extra 2: Se ha añadido color a la consola para facilitar la lectura del mapa.
- Extra 3: Se ha añadido como coste la distancia euclídea hacia la meta. 
- Extra 4: Se ha añadido el paso de argumentos al código para escoger entre el environment csv-v0 o csv-pygame-v0.
- Extra 5: Se puede escoger entre dos algoritmos implementados, el primer algoritmo es Q-Learning y el segundo es Value Iteration Algorithm.
- Extra 6: Se ha añadido la lectura por consola para introducir las coordenadas de inicio y meta.
- Extra 7: Se han creado otros mapas adicionales que se escogen por entrada en consola. 
- Extra 8: Se ha añadido el movimiento en diagonal, no solo lateralmente.
- Extra 9: Se puede pasar por argumentos el número de episodios para los algoritmos.
- Extra 10: Se ha subido demostrativo a [YouTube](https://youtu.be/_EgTOtM6OyY).

Para ejecutar el código:
```
$ python gym-csv-loop.py --algorithm vai --use_pygame true --episodes 650
```