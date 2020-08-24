[click here for English version](https://onesait-git.cwbyminsait.com/data/generic-classifier-pipeline/blob/dev/readme-en.md)

# Clasificador genérico

### Este proyecto contiene: <p>

* modulos con fucniones para cargar y transformar datos procedentes de un archivo csv o de una ontología Onesait
* herramientas de visualización
* análisis exploratorio del dataset y metodologías de preprocesado
* validación cruzada en búsqueda exhaustiva con cada posible combinación de modelo-hiperparámetros   
* implementación de tests unitarios 

## Paquete de extracción de datos:

El paquete llamado 'dataset_elt' incluye una forma sencilla de cargar el dataset requerido, tanto desde una ontología Onesait como de un archivo csv. <p>
Si la extracción es desde una ontología Onesait, el usuario debe especificar:<p>

* iot_client_host: servidor donde la ontología está almacenada
* iot_client_name: nombre del cliente creado para poder acceder a dicha ontología 
* iot_client_token: token de acceso

Si el acceso es a un archivo csv:

* dataset_location: ruta donde se encuentra almacenado el archivo csv

## Visualizaciones de los datos

El paquete llamado 'dataset_plots' tiene varias utilidades para representar el dataset extraído, como parte del analísis exploratorio. La visualización de datos es una herramienta útil para adquirir una primera impresión de los datos antes de implementar cambios. Como tal, debe tomarse con cautela, ya que a veces puede confundir si dichos datos no se exponen a análisis más específicos.

## EDA (análisis exploratorio de los datos) 

Como ya se ha mencionado, podemos implementar un análisis exploratorio en dicho dataset; con esto, podemos automatizar la selección de atributos mediante correlaciones entre atributos y otros aspectos a considerar. Un ejemplo de esto se puede encontrar en '\dataset_EDA\eda_reports', con un reporte en formato HTML. Con este útil e interactivo reporte en formato, podemos encontrar valores estadísticos descriptivos, matrices de correlación e incluso automatizar los atributos seleccionados basados en un umbral indicado (por ejemplo, eliminación de uno de los atributos perteneciente a un par que presenten una correlación de más del 90%): 

![Alt text](\readme_files/EDA_2_opt.png "EDA example 1")
![Alt text](\readme_files/EDA_opt.png "EDA example 2")

## Preprocesado

Con el paquete 'dataset_preprocessing' podemos buscar valores ausentes, imputarlos con las estrategia deseada, o buscar valores anómalos...

## Modelización

Una vez tenemos el dataset preparado para modelarlo, este paquete 'modelling' nos permite implementar un grid search con la estrategia de validación cruzada para la selección del modelo y sus hiperparámetros. El usuario puede indicar la lista de modelos que quiera intentar aplicar, con la correspondiente lista de hiperparámetros para cada modelo.
Como resultado, obtendremos una tabla con la mejor combinación de hiperparámetros para cada uno de los tipos de modelos entrenados.
La lista introducida debe estar implementada hoy día en scikit-learn, aunque la misma lógica podrá ser aplicada a otras librerías.

## Validación

Finalmente, el mejor modelo es seleccionado basado en la métrica indicada para dicha decisión, de entre los modelos devueltos por el paso de modelado. 
Un ejemplo del resultado de la validación cruzada, para cada modelo, se presenta a continuación, mostrando un mean_test_recall de 0.98 para el mejor clasificador:

![Alt text](\readme_files/cross_validation_results_opt.jpg "Validation results")

Para este ejemplo, ya que hay un gran desbalanceo entre los casos de lluvia/no lluvia, y considerando que un predictor del tiempo debería ser principalmente preciso en los casos de lluvia real, se ha escogido la sensibilidad como métrica principal de validación, por tanto en detectar los verdaderos positivos (la misma lógica aplica a la detección de enfermedades). Más info sobre esta métrica: https://es.wikipedia.org/wiki/Precisi%C3%B3n_y_exhaustividad#Exhaustividad

## Tests unitario

Como cualquier proyecto de software, la fase de tests es esencial para entregar un producto fiable; con frameworks como unittest, py.test... podemos implementar tanto tests de código y funcionales como queramos. Un ejemplo de algunos tests de este proyecto son:

![Alt text](\readme_files/tests_opt.jpg "Tests")