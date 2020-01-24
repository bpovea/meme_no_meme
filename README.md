# Meme / No Meme

Este proyecto consiste clasificar imágenes usando una red neuronal convolucional (CNN). Para ello utilizamos la arquitectura de LeNet ya que es una de las primeras usadas para clasificar imágenes cuyos resultados son óptimos.

- Dos conjuntos de capas convolucionales, de activación y agrupación. Seguidas de dos capas fully connected y otra de activación. Finalmente un clasificador con función softmax
- 25 épocas
- batch size de 32
- Imágenes pre-procesadas 28x28 - grises


## Instalación de dependencias

### Creación de entorno virtual [opcional]

Se recomienda usar un entorno virtual para instalar las dependencias del proyecto

```bash
virtualenv env --python python3
```
### Dependencias

El listado de dependencias se encuentra en el archivo requirements.txt, para instalarlo facilmente usaremos pip

```bash
pip install -r requirements.txt
```
### Directorio de archivos

Es necesario tener la siguiente estructura de archivos para poder entrenar la red:

    .
    ├── ...
    ├── images
    │   ├── meme
    │   │   └── ...             # Aquí deben estar las imágenes de memes
    │   ├── not_meme
    │       └── ...             # Aquí deben estar las imágenes que no son memes
    └── ...

## Usage - Entrenamiento

Para realizar el entrenamiento de la red se debe ejecutar el script train_network.py


```bash
python train_network.py
```

Este script se encargará de cargar las imágenes, procesarlas, entrenar la cnn y crear plots de los resultados del entrenamiento. Finalmente como resultado creará un archivo llamado meme_not_meme.model el cual contendrá nuestra red ya entrenada, adicionalmente creará los archivos plot.png y confusion.png que contendrán los resultados del entrenamiento.

En este proyecto ya se encuentra un archivo meme_not_meme.model el cual contiene ya nuestro resultado actual del entrenamiento.


Tambien hay un archivo meme_no_meme.ipynb el cual puede ser revisado usando Jupyter notebook o abriendolo desde el repositorio en github.com.


## Usage - análisis de imágen

Para analizar un archivo y ver el porcentaje de coincidencia si es meme o no lo es, utilizamos el archivo test_network.py ejecutandolo con el siguiente comando

```bash
python test_network.py --model meme_not_meme.model --image ./path/to/image.jpg
```
Como resultado tendremos un plot que indicará el porcentaje de coincidencia; para cerrarlo presione 0.


## Usage - interfaz

Para ejecutar la interfaz y clasificar toda una carpeta de imágenes, use el archivo interface5.py

```bash
python interface5.py
```
Llenar el cuadro de texto con la ruta completa de la carpeta a ser analizada y dar click en el botón process. Finalmente el script moverá las imágenes que considere memes a una subcarpeta que creará con el nombre de "memes".

### Antes
    .
    ├── ...
    ├── galeria
    │   └── ...                 # Aquí se encuentran todas las imágenes mezcladas
    └── ...
### Después
    .
    ├── ...
    ├── galeria
    │   ├── ...                 # Aquí se encuentran los no memes
    │   └── memes
    │       └── ...             # Aquí se encuentran los memes
    └── ...

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)