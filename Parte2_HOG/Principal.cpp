#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <iomanip> // Para formatear la matriz de confusión

using namespace cv;
using namespace std;

// Función para calcular el HOG de una imagen
vector<float> obtenerHOG(const Mat& imagen) {
    cv::HOGDescriptor hog;

    // Establecer los parámetros del HOGDescriptor
    hog.winSize = Size(64, 128);
    hog.blockSize = Size(16, 16);
    hog.blockStride = Size(8, 8);
    hog.cellSize = Size(8, 8);
    hog.nbins = 9;

    vector<float> descriptor;
    hog.compute(imagen, descriptor);
    return descriptor;
}

// Función para leer las imágenes de una carpeta
vector<Mat> leerImagenes(const string& ruta) {
    vector<Mat> imagenes;
    vector<string> nombres_imagenes;
    glob(ruta, nombres_imagenes);

    for (const auto& nombre_imagen : nombres_imagenes) {
        Mat imagen = imread(nombre_imagen);
        if (!imagen.empty()) {
            imagenes.push_back(imagen);
        }
    }

    return imagenes;
}

// Función para imprimir la matriz de confusión
void imprimirMatrizConfusion(const Mat& matriz_confusion, int num_clases) {
    for (int i = 0; i < num_clases; ++i) {
        for (int j = 0; j < num_clases; ++j) {
            cout << setw(5) << matriz_confusion.at<int>(i, j) << " ";
        }
        cout << endl;
    }
}

int main() {
    vector<string> carpetas = {"images/android", "images/batman", "images/chrome", "images/facebook", "images/instagram"};
    vector<Mat> imagenes_entrenamiento;
    vector<int> etiquetas_entrenamiento;

    for (int i = 0; i < carpetas.size(); ++i) {
        string ruta = "./" + carpetas[i] + "/*";
        vector<Mat> imagenes = leerImagenes(ruta);
        for (const auto& imagen : imagenes) {
            Mat imagen_redimensionada;

            // Preprocesamiento de la imagen
            resize(imagen, imagen_redimensionada, Size(64, 128));
            cvtColor(imagen_redimensionada, imagen_redimensionada, COLOR_BGR2GRAY); // Convertir a escala de grises
            equalizeHist(imagen_redimensionada, imagen_redimensionada); // Ecualización del histograma
            GaussianBlur(imagen_redimensionada, imagen_redimensionada, Size(3, 3), 0); // Suavizado para reducir ruido

            vector<float> descriptor = obtenerHOG(imagen_redimensionada);
            imagenes_entrenamiento.push_back(imagen_redimensionada);
            etiquetas_entrenamiento.push_back(i);
        }
    }

    Ptr<ml::SVM> svm = ml::SVM::create();
    svm->setKernel(ml::SVM::RBF);  // Usar un kernel RBF para un mejor ajuste
    svm->setType(ml::SVM::C_SVC);
    svm->setC(1);  // Ajustar este parámetro para mejorar la generalización

    Mat datos_entrenamiento(imagenes_entrenamiento.size(), 3780, CV_32F);
    for (size_t i = 0; i < imagenes_entrenamiento.size(); ++i) {
        vector<float> descriptor = obtenerHOG(imagenes_entrenamiento[i]);
        for (size_t j = 0; j < descriptor.size(); ++j) {
            datos_entrenamiento.at<float>(i, j) = descriptor[j];
        }
    }

    Ptr<ml::TrainData> trainData = ml::TrainData::create(datos_entrenamiento, ml::SampleTypes::ROW_SAMPLE, etiquetas_entrenamiento);
    svm->train(trainData);
    svm->save("svm_logo.xml");

    cout << "Clasificador entrenado y guardado como 'svm_logo.xml'." << endl;

    // Crear una matriz de confusión
    int num_clases = carpetas.size();
    Mat matriz_confusion = Mat::zeros(num_clases, num_clases, CV_32S);  // Matriz de confusión inicializada a 0

    // Clasificar las imágenes de entrenamiento para crear la matriz de confusión
    for (size_t i = 0; i < imagenes_entrenamiento.size(); ++i) {
        vector<float> descriptor = obtenerHOG(imagenes_entrenamiento[i]);
        Mat descriptor_mat(1, descriptor.size(), CV_32F);
        for (size_t j = 0; j < descriptor.size(); ++j) {
            descriptor_mat.at<float>(0, j) = descriptor[j];
        }

        // Predecir la etiqueta
        int etiqueta_predicha = svm->predict(descriptor_mat);
        int etiqueta_real = etiquetas_entrenamiento[i];

        // Actualizar la matriz de confusión
        matriz_confusion.at<int>(etiqueta_real, etiqueta_predicha)++;
    }

    // Imprimir la matriz de confusión
    cout << "Matriz de Confusión:" << endl;
    imprimirMatrizConfusion(matriz_confusion, num_clases);

    // Cargar una nueva imagen para predecir
    string ruta_imagen_prueba = "images/chrome/C_chrome_1.png"; // Ruta de la imagen de prueba
    Mat imagen_prueba = imread(ruta_imagen_prueba);

    if (!imagen_prueba.empty()) {
        Mat imagen_redimensionada;

        // Preprocesar la imagen de prueba
        resize(imagen_prueba, imagen_redimensionada, Size(64, 128));
        cvtColor(imagen_redimensionada, imagen_redimensionada, COLOR_BGR2GRAY); // Convertir a escala de grises
        equalizeHist(imagen_redimensionada, imagen_redimensionada); // Ecualización del histograma
        GaussianBlur(imagen_redimensionada, imagen_redimensionada, Size(3, 3), 0); // Suavizado para reducir ruido

        vector<float> descriptor = obtenerHOG(imagen_redimensionada);
        Mat descriptor_mat(1, descriptor.size(), CV_32F);
        for (size_t i = 0; i < descriptor.size(); ++i) {
            descriptor_mat.at<float>(0, i) = descriptor[i];
        }

        // Obtener la predicción
        int etiqueta_predicha = svm->predict(descriptor_mat); // Predicción directa
        cout << "El logo predicho es: " << carpetas[etiqueta_predicha] << endl;

        // Mostrar la imagen de prueba con el logo predicho
        imshow("Imagen Predicha", imagen_prueba);
        waitKey(0); // Esperar hasta que el usuario presione una tecla para cerrar la ventana

    } else {
        cout << "No se pudo cargar la imagen de prueba." << endl;
    }

    return 0;
}
