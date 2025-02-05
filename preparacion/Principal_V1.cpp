// Principal_V1.cpp

// Definir ImageMatrix para que se interprete como cv::Mat
#define ImageMatrix cv::Mat

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include "zernike.h"   // Ubicado en: /home/mateo/Aplicaciones/Librerias/opencv/pychrm/src/textures/zernike/zernike.h

using namespace cv;
using namespace std;

// Estructura para almacenar los momentos de cada figura
struct Figura {
    string etiqueta;
    vector<double> momentos;
};

/// -------------------------------------------------------------------------
/// Función: computeZernikeMomentsWrapper
/// Función *wrapper* que invoca la función de la librería (mb_zernike2D) para calcular
/// los momentos de Zernike a partir de una imagen. Se preprocesa la imagen (conversión a gris,
/// umbralización) y se calcula un radio R para la normalización.
/// Parámetros:
///   - imagen: imagen de entrada (cv::Mat)
///   - order: orden de los momentos (por ejemplo, 4)
/// Retorna:
///   - vector<double> con los momentos de Zernike.
vector<double> computeZernikeMomentsWrapper(const Mat &imagen, int order) {
    // Convertir la imagen a escala de grises
    Mat gris;
    if (imagen.channels() == 3)
        cvtColor(imagen, gris, COLOR_BGR2GRAY);
    else
        gris = imagen.clone();
    
    // Umbralizar para obtener una imagen binaria.
    // Ajusta THRESH_BINARY o THRESH_BINARY_INV según tus imágenes.
    Mat binaria;
    threshold(gris, binaria, 128, 255, THRESH_BINARY_INV);
    
    // Definir el radio R como la mitad del mínimo de ancho y alto
    double R = min(binaria.cols, binaria.rows) / 2.0;
    
    // Reservar un buffer para los momentos (ajusta buffer_size según sea necesario)
    const int buffer_size = 1000;
    double *zvalues = new double[buffer_size];
    long output_size = 0;
    
    // Llamar a la función de la librería.
    // La firma es:
    // void mb_zernike2D(const ImageMatrix &I, double D, double R, double *zvalues, long *output_size);
    // Donde D es el orden (convertido a double).
    mb_zernike2D(binaria, static_cast<double>(order), R, zvalues, &output_size);
    
    // Copiar los resultados a un vector y liberar el buffer
    vector<double> momentos(zvalues, zvalues + output_size);
    delete [] zvalues;
    
    return momentos;
}

/// -------------------------------------------------------------------------
/// Función: calcularMomentosZernike
/// Calcula los momentos de Zernike a partir de una imagen usando la función wrapper.
vector<double> calcularMomentosZernike(const Mat& imagen, int order = 4) {
    return computeZernikeMomentsWrapper(imagen, order);
}

/// -------------------------------------------------------------------------
/// Función: cargarDatasetZernike
/// Carga el dataset de momentos de Zernike desde un archivo CSV. Cada línea del CSV debe tener el formato:
///   Clase,zm1,zm2,...,zmN
vector<Figura> cargarDatasetZernike(const string& archivoCSV) {
    vector<Figura> dataset;
    ifstream archivo(archivoCSV);
    if (!archivo) {
        cout << "Error al abrir el archivo " << archivoCSV << endl;
        return dataset;
    }

    string linea;
    getline(archivo, linea); // Saltar la cabecera, si existe

    while (getline(archivo, linea)) {
        stringstream ss(linea);
        string etiqueta;
        getline(ss, etiqueta, ',');

        vector<double> momentos;
        string token;
        while (getline(ss, token, ',')) {
            try {
                momentos.push_back(stod(token));
            } catch (...) {
                // Ignorar errores de conversión
            }
        }
        dataset.push_back({etiqueta, momentos});
    }
    archivo.close();
    return dataset;
}

/// -------------------------------------------------------------------------
/// Función: distanciaEuclidea
/// Calcula la distancia euclídea entre dos vectores.
double distanciaEuclidea(const vector<double>& a, const vector<double>& b) {
    if (a.size() != b.size())
        return 1e9;  // Evitar errores en caso de tamaños distintos

    double suma = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        suma += pow(a[i] - b[i], 2);
    }
    return sqrt(suma);
}

/// -------------------------------------------------------------------------
/// Función: clasificarImagen
/// Clasifica una imagen comparando sus momentos de Zernike con los del dataset.
string clasificarImagen(const Mat& imagen, const vector<Figura>& dataset, int order = 4) {
    vector<double> momentosFigura = calcularMomentosZernike(imagen, order);
    if (momentosFigura.empty())
        return "No se pudo calcular";

    string clasePredicha = "Desconocido";
    double minDistancia = 1e9;
    for (const auto& figura : dataset) {
        double dist = distanciaEuclidea(momentosFigura, figura.momentos);
        if (dist < minDistancia) {
            minDistancia = dist;
            clasePredicha = figura.etiqueta;
        }
    }
    return clasePredicha;
}

/// -------------------------------------------------------------------------
/// Función principal
int main() {
    // 1. Ruta fija de la imagen a clasificar (modifícala según corresponda)
    string rutaImagen = "/home/mateo/Escritorio/U/Vision_Computador/Unidad_3/Practicas/Practica_3.1/preparacion/testing/ejemplo.png";
    Mat imagen = imread(rutaImagen, IMREAD_COLOR);
    if (imagen.empty()) {
        cout << "No se pudo cargar la imagen." << endl;
        return -1;
    }
    
    // 2. Cargar el dataset de momentos de Zernike desde el CSV
    string rutaCSV = "dataset_zernike.csv";  // Asegúrate de que el CSV tenga el formato correcto
    vector<Figura> dataset = cargarDatasetZernike(rutaCSV);
    if (dataset.empty()) {
        cout << "El dataset está vacío o no se pudo cargar." << endl;
        return -1;
    }
    
    // 3. Clasificar la imagen usando momentos de Zernike (orden 4, por ejemplo)
    int order = 4;  // Puedes experimentar con otros órdenes
    string resultado = clasificarImagen(imagen, dataset, order);
    cout << "La imagen se clasifica como: " << resultado << endl;
    
    // 4. Mostrar el resultado sobre la imagen original
    putText(imagen, resultado, Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 0, 255), 2);
    imshow("Imagen Clasificada", imagen);
    waitKey(0);
    
    return 0;
}
