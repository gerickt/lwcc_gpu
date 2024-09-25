Aquí tienes la traducción del README con los créditos correspondientes:

---

# LWCC: Una librería ligera para el conteo de multitudes en Python

![](https://img.shields.io/badge/state-of%20the%20art-orange) ![](https://img.shields.io/github/license/tersekmatija/lwcc?label=license) [![](https://pepy.tech/badge/lwcc)](https://pepy.tech/project/lwcc) ![](https://img.shields.io/github/stars/tersekmatija/lwcc) ![](https://img.shields.io/pypi/v/lwcc?color=pink)

![](https://raw.githubusercontent.com/tersekmatija/lwcc/master/imgs/lwcc_header_gif.gif)

LWCC es un framework ligero para el conteo de multitudes en Python. Envuelve cuatro modelos de vanguardia, todos basados en redes neuronales convolucionales: [`CSRNet`](https://github.com/leeyeehoo/CSRNet-pytorch), [`Conteo de multitudes Bayesiano`](https://github.com/ZhihengCV/Bayesian-Crowd-Counting), [`DM-Count`](https://github.com/cvlab-stonybrook/DM-Count), y [`SFANet`](https://github.com/pxq0312/SFANet-crowd-counting). La librería está basada en PyTorch.

## Instalación

La forma más sencilla de instalar la librería LWCC y sus requisitos es utilizando el gestor de paquetes [pip](https://pip.pypa.io/en/stable/).

```python
pip install git+https://github.com/gerickt/lwcc_gpu.git
```

## Uso
Puedes importar la librería y utilizar sus funcionalidades de la siguiente manera:

```python
from lwcc import LWCC
```

### Estimación de conteo
La forma más directa de usar la librería:

```python
img = "ruta/a/la/imagen"
count = LWCC.get_count(img)
```

Esto utiliza CSRNet preentrenado en SHA (por defecto). Puedes elegir un modelo diferente preentrenado en otro conjunto de datos usando:

```python
count = LWCC.get_count(img, model_name = "DM-Count", model_weights = "SHB")
```

El resultado es un valor flotante con el conteo predicho.

### Imágenes grandes

**Nota**: Por defecto, todas las imágenes se redimensionan para que el lado más largo sea menor a 1000 px, manteniendo la proporción. De lo contrario, los modelos pueden rendir peor en imágenes grandes con multitudes dispersas (contando patrones en camisetas, vestidos). Si estás estimando multitudes densas, se recomienda que desactives el redimensionamiento estableciendo *resize_img* a *False*. La llamada se vería así:

```python
count = LWCC.get_count(img, model_name = "DM-Count", model_weights = "SHB", resize_img = False)
```

### Múltiples imágenes

La librería permite la predicción de conteo para múltiples imágenes en una sola llamada a *get_count*. Simplemente puedes pasar una lista de rutas de imágenes:

```python
img1 = "ruta/a/la/imagen1"
img2 = "ruta/a/la/imagen2"
count = LWCC.get_count([img1, img2])
```

El resultado será un diccionario con pares *nombre_de_imagen : conteo_de_imagen*:
![result](https://raw.githubusercontent.com/tersekmatija/lwcc/master/imgs/result.png)

### Mapa de densidad

También puedes solicitar un mapa de densidad estableciendo el parámetro *return_density = True*. El resultado será una tupla *(conteo, mapa_de_densidad)*, donde *mapa_de_densidad* es un arreglo 2D con las densidades predichas. El tamaño del arreglo es menor que el de la imagen de entrada y depende del modelo.

```python
import matplotlib.pyplot as plt

count, density = LWCC.get_count(img, return_density = True)

plt.imshow(density)
plt.show()
```

![result_density](https://raw.githubusercontent.com/tersekmatija/lwcc/master/imgs/result_density.png)

Esto también funciona para múltiples imágenes (lista de rutas de imágenes como entrada). El resultado será una tupla de dos diccionarios, donde el primer diccionario es igual al anterior (pares de *nombre_de_imagen : conteo_de_imagen*) y el segundo diccionario contiene pares de *nombre_de_imagen : mapa_de_densidad*.

### Cargando el modelo

También puedes acceder directamente a los modelos de PyTorch cargándolos primero con el método *load_model*.

```python
model = LWCC.load_model(model_name = "DM-Count", model_weights = "SHA")
```

El *modelo* cargado es un modelo de PyTorch y puedes acceder a sus pesos como con cualquier otro modelo de PyTorch.

Puedes usarlo para inferencia como:

```python
 count = LWCC.get_count(img, model = model)
```

## Modelos

LWCC actualmente ofrece 4 modelos (CSRNet, Conteo de multitudes Bayesiano, DM-Count, SFANet) preentrenados en los conjuntos de datos [Shanghai A](https://ieeexplore.ieee.org/document/7780439), [Shanghai B](https://ieeexplore.ieee.org/document/7780439) y [UCF-QNRF](https://www.crcv.ucf.edu/data/ucf-qnrf/). La siguiente tabla muestra el nombre del modelo y los resultados de MAE / MSE de los modelos preentrenados en los conjuntos de prueba disponibles.

|   Nombre del modelo |      SHA       |      SHB      |      QNRF       |
| ------------------: | :------------: | :-----------: | :-------------: |
|   **CSRNet**        | 75.44 / 113.55 | 11.27 / 19.32 | *No disponible* |
|      **Bay**        | 66.92 / 112.07 | 8.27 / 13.56  | 90.43 / 161.41  |
| **DM-Count**        | 61.39 / 98.56  | 7.68 / 12.66  | 88.97 / 154.11  |
|   **SFANet**        | *No disponible* | 7.05 / 12.18  | *No disponible* |

Opciones válidas para *model_name* están escritas en la primera columna e incluyen: `CSRNet`, `Bay`, `DM-Count`, y `SFANet`.
Opciones válidas para *model_weights* están escritas en la primera fila e incluyen: `SHA`, `SHB` y `QNRF`.

**Nota**: No todos los *model_weights* son compatibles con todos los *model_names*. Revisa la tabla anterior para combinaciones posibles.

## ¿Cómo funciona?

El objetivo de los métodos de conteo de multitudes es determinar el número de personas presentes en un área particular. Existen muchos enfoques (detección, regresión, enfoques basados en densidad), sin embargo, desde 2015 se han propuesto muchos enfoques basados en redes neuronales convolucionales (CNN). La idea básica de estos enfoques es que intentan predecir el mapa de densidad a partir de la imagen de entrada e inferir el conteo a partir de él. Estos modelos difieren en el uso de diferentes backbones, funciones de pérdida, mapas adicionales, etc. Si te interesa un algoritmo específico, te invitamos a leer el artículo relacionado con ese modelo.

## FAQ - Preguntas frecuentes

### ¿Puedo ver más ejemplos de LWCC en acción?

Sí, puedes encontrar algunos ejemplos en [Examples.ipynb](https://github.com/tersekmatija/lwcc/blob/master/tests/Examples.ipynb).

### ¿Qué tan precisos son los modelos?

Puedes ver el error absoluto medio (MAE) y el error cuadrático medio (MSE) de los modelos preentrenados en los conjuntos de prueba en la sección [modelos](#models). Recomendamos los modelos preentrenados en SHA o QNRF para multitudes densas, y SHB para multitudes dispersas.

### ¿Hay soporte para GPU?

No, actualmente no se admite el soporte para GPU, pero está planificado para una futura versión.

### ¿Puedo cargar pesos personalizados?

El soporte completo para cargar pesos personalizados no está disponible actualmente, pero está planeado para una futura versión.

### ¿Puedo entrenar los modelos yo mismo?

La librería no admite entrenamiento, solo inferencia.

### ¿Por qué mis resultados son malos?

Esto puede depender del modelo que uses, el tamaño de la imagen, la densidad o el tipo de multitud, o los pesos que estés usando. Por ejemplo, los modelos a menudo cometen errores en imágenes de retratos grupales, ya que están entrenados en imágenes que contienen multitudes en calles, conciertos, etc. Usar los pesos de `SHA` en multitudes relativamente dispersas también podría dar resultados muy incorrectos. Por otro lado, `SHB` podría funcionar mejor, ya que los pesos fueron entrenados en el conjunto de datos de Shanghai B, que contiene imágenes con multitudes relativamente dispersas. Usar imágenes de alta calidad con multitudes dispersas también puede dar malos resultados, ya que los algoritmos podrían confundir algunas texturas de ropa con una multitud.

Como regla general, deberías usar `SHB` si planeas estimar el número de personas en imágenes con multitudes dispersas, y `SHA

` o `QNRF` para imágenes con multitudes densas. Ten en cuenta que los algoritmos actuales predicen la densidad, y aún podrían cometer algunos errores. Te invitamos a probar diferentes combinaciones de modelos y pesos para ver cuál funciona mejor para tu problema.

## Soporte

Si te gusta la librería, ¡por favor muéstranos tu apoyo con una ⭐️ al proyecto!

Si deseas incluir tu propio modelo de conteo de multitudes, contáctanos a (*matijatersek@protonmail.com* o *masika.kljun@gmail.com*).

### Stargazers

[![Stargazers repo roster for @tersekmatija/lwcc](https://reporoster.com/stars/tersekmatija/lwcc)](https://github.com/tersekmatija/lwcc/stargazers)

## Citación

Esta librería es el resultado de una investigación de modelos de conteo de multitudes con CNN realizada por Matija Teršek y Maša Kljun. Aunque el artículo aún no ha sido publicado, por favor proporciona el enlace a este repositorio de GitHub si usas LWCC en tu investigación.

## Licencia

Esta librería está licenciada bajo la licencia MIT (ver [LICENSE](https://github.com/tersekmatija/lwcc/blob/master/LICENSE)). Las licencias de los modelos envueltos en la librería se heredan, dependiendo del modelo que utilices ([`CSRNet`](https://github.com/leeyeehoo/CSRNet-pytorch), [`Conteo de multitudes Bayesiano`](https://github.com/ZhihengCV/Bayesian-Crowd-Counting), [`DM-Count`](https://github.com/cvlab-stonybrook/DM-Count), y [`SFANet`](https://github.com/pxq0312/SFANet-crowd-counting)).

---

**Fork realizado por Gerick Toro para dar soporte a CUDA y permitir el procesamiento de imágenes con GPU.**
