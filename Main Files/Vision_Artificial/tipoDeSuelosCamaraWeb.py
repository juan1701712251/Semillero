import numpy as np
from PIL import Image
import easygui as eg
import cv2
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.models import load_model
from keras.models import model_from_json
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import img_to_array
import threading
import tensorflow as tf


############## POSIBLES CLASES ##################
from scipy.stats._multivariate import random_correlation_gen


class VisionArtificial():
    def __init__(self):
        self.dic_clas = {0: 'AnnualCrop',
                    1: 'Forest',
                    2: 'HerbaceousVegetation',
                    3: 'Highway',
                    4: 'Industrial',
                    5: 'Pasture',
                    6: 'PermanentCrop',
                    7: 'Residential',
                    8: 'River',
                    9: 'SeaLake'}

        self.dic_clas_spanish = {0: 'Cultivos o Sabanas',
                            1: 'Bosque',
                            2: 'Vegetacion herbacea',
                            3: 'Autopista',
                            4: 'Industrial',
                            5: 'Pasto',
                            6: 'Cultivo permanente',
                            7: 'Residencial',
                            8: 'Rio',
                            9: 'Mar'}
        self.loaded_model = None
        self.loadNeuralNetwork()
        self.WebCamWindow = None
        self.enEjecucion=True

    def loadNeuralNetwork(self):

        # cargar json y crear el modelo
        json_file = open('Vision_Artificial/tipodesuelos.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        self.loaded_model = model_from_json(loaded_model_json)

        #necesario
        global graph
        graph = tf.get_default_graph()
        global sess
        sess = tf.Session()
        set_session(sess)

        # Cargar pesos al nuevo modelo
        self.loaded_model.load_weights("Vision_Artificial/tipodesuelos.h5")

        #print("Cargado modelo desde disco.")

        #Compilamos el modelo
        self.loaded_model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])


#Elegimos una imagen

    def predictImage(self):
        extension = ["*.jpg", "*.png"]
        imagen_url = eg.fileopenbox(msg="Abrir archivo",
                                 title="Control: fileopenbox",
                                 default='',
                                 filetypes=extension)


        if imagen_url:
            return self.predict_image_url(imagen_url)
        return 'None'

    def predict_image_url(self,imagen_url):
        # Guardamos la imagen dentro de nuestro arreglo que sera la entrada de nuestra red neuronal
        image = Image.open(str(imagen_url))
        image_resize = image.resize((64,64))
        im_array = np.array(image_resize, dtype='float32')
        datos_entrada = (np.array([im_array])) / 255
        print(str(datos_entrada))
        # El modelo predice la entrada y genera un arreglo con la probabilidad de cada clase de imagen
        prob_pred_test = self.loaded_model.predict(datos_entrada)
        #DEPRICATED ->>>prob_pred_test = self.loaded_model._make_predict_function(datos_entrada)
        print(prob_pred_test)
        # Escogemos la probabilidad mayor y la mostramos
        predicciones_test = [self.dic_clas_spanish[np.argmax(prob)] for prob in prob_pred_test]

        return predicciones_test

    def predictWebCam(self,puertoCamara):
        self.video_threading = threading.Thread(target=self.video, args=puertoCamara)
        self.video_threading.start()

    def video(self,puertoCamara):
        self.WebCamWindow = cv2.VideoCapture(int(puertoCamara))
        while (1):
            ret,frame = self.WebCamWindow.read()
            img_resize = self.resizeImage(frame)
            cv2.imwrite('frame-resize.jpg', img_resize)

            '''
            #creamos una copia de la imagen
            copy_frame = frame.copy()
            #Convertimos el frame en un vector de datos
            #img_proccess = preprocess_input(copy_frame)
            #img_resize = self.resizeImage(img_proccess)
            #im_array = np.array(img_resize,dtype='float32')
            
            '''

            with graph.as_default():
                set_session(sess)
                prediccion = self.predict_image_url('frame-resize.jpg')
            print(prediccion)

            mensaje = "Prediccion zona-->   " + str(prediccion[0])
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(frame, mensaje, (10, 40), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


            # Se muestra en la ventana
            cv2.imshow('Video_Prediccion', frame)

            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
        self.WebCamWindow.release()
        cv2.destroyAllWindows()


    def resizeImage(self,image):
        print(image.shape)
        newImage = cv2.resize(image,(64,64))
        print(newImage.shape)
        cv2.imshow('Video_Redimension',newImage)
        return newImage

