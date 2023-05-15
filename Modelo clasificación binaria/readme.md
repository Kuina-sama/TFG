Este modelo plantea el problema como una clasificación binaria.
Es decir, la capa de salida tiene una única neurona y se usa una sigmoide: Si la salida es 1 clasificamos como hombre,
si es 0, asumimos que "no hombre" equivale a mujer.

Se ha descartado temporalmente esta solución por dar todas las métricas igual (en teoría se supone que dan igual por ser clases mutuamente excluyentes.

Para un mejor análisis se vuelve a pasar a una softmax con DOS neuronas de salida, quedándonos con la que de mayor probabilidad.

Explicación de diferencia entre softmax y sigmoid:
https://towardsdatascience.com/sigmoid-and-softmax-functions-in-5-minutes-f516c80ea1f9#:~:text=Sigmoid%20is%20used%20for%20binary,extension%20of%20the%20Sigmoid%20function.

Volver a la aproximación de softmax permite luego ampliar a clasificación con 3 clases en las que tengamos la clase unknown y esta no sea sustituida por male o female.