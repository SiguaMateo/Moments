#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <filesystem>
#include <cmath>
#include <fstream>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

// Función para calcular la distancia euclidiana entre dos vectores
double calcularDistancia(const vector<double>& a, const vector<double>& b) {
    double distancia = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        distancia += abs(a[i] - b[i]);
    }
    return distancia;
}

// Función para normalizar un vector (normalización Z-score)
vector<double> normalizar(const vector<double>& vec) {
    double suma = 0.0;
    for (double v : vec) {
        suma += v;
    }
    double media = suma / vec.size();

    double varianza = 0.0;
    for (double v : vec) {
        varianza += pow(v - media, 2);
    }
    double desviacion = sqrt(varianza / vec.size());

    vector<double> normalizado;
    for (double v : vec) {
        normalizado.push_back((v - media) / desviacion);  // Normalización Z-score
    }
    return normalizado;
}

// Función para calcular los momentos de Hu de una imagen binaria
vector<double> calcularMomentosHu(const Mat& imagen) {
    Moments moments = cv::moments(imagen, true);
    double huMoments[7];
    HuMoments(moments, huMoments);

    // Convertir los momentos a un vector
    return vector<double>(huMoments, huMoments + 7);
}

// Función para aplicar preprocesamiento adicional (filtros, bordes, contraste)
Mat preprocesarImagen(const Mat& img) {
    Mat gray, blurred, edges;

    // Convertir a escala de grises
    cvtColor(img, gray, COLOR_BGR2GRAY);

    // Suavizado para reducir el ruido
    GaussianBlur(gray, blurred, Size(5, 5), 0);

    // Detección de bordes usando Canny
    Canny(blurred, edges, 100, 200);

    return edges;
}

// Función para procesar una carpeta y calcular los momentos promedio
vector<double> calcularPromedioMomentos(const string& carpeta, const string& clase) {
    vector<vector<double>> momentosClase;
    for (const auto& entrada : fs::directory_iterator(carpeta)) {
        Mat img = imread(entrada.path().string(), IMREAD_COLOR);
        if (img.empty()) continue;

        // Aplicar preprocesamiento adicional
        Mat imgPreprocesada = preprocesarImagen(img);

        // Calcular los momentos de Hu de la imagen preprocesada
        momentosClase.push_back(calcularMomentosHu(imgPreprocesada));
    }

    // Calcular el promedio de los momentos
    vector<double> promedio(7, 0.0);
    for (const auto& momentos : momentosClase) {
        for (size_t i = 0; i < 7; i++) {
            promedio[i] += momentos[i];
        }
    }

    for (size_t i = 0; i < 7; i++) {
        promedio[i] /= momentosClase.size();
    }

    //promedio = normalizar(promedio);
    
    // Guardar los momentos calculados en un archivo CSV
    ofstream archivo("momentos_hu.csv", ios::app); // Abre el archivo en modo append
    if (archivo.is_open()) {
        archivo << clase; // Escribir el nombre de la clase
        for (size_t i = 0; i < 7; i++) {
            archivo << "," << promedio[i]; // Escribir los momentos
        }
        archivo << endl; // Nueva línea después de cada entrada
        archivo.close();
    } else {
        cerr << "No se pudo abrir el archivo momentos_hu.csv." << endl;
    }

    return promedio;
}

// Función para leer los momentos promedio desde un archivo CSV
vector<pair<string, vector<double>>> leerMomentosDesdeCSV(const string& archivoCSV) {
    vector<pair<string, vector<double>>> momentos;
    ifstream archivo(archivoCSV);
    if (!archivo.is_open()) {
        cerr << "No se pudo abrir el archivo " << archivoCSV << endl;
        return momentos;
    }

    string linea;
    while (getline(archivo, linea)) {
        stringstream ss(linea);
        string clase;
        getline(ss, clase, ',');

        vector<double> momentosClase;
        string valor;
        while (getline(ss, valor, ',')) {
            momentosClase.push_back(stod(valor));
        }
        momentos.push_back({clase, momentosClase});
    }
    archivo.close();
    return momentos;
}

int main() {
    // Directorios del dataset
    string carpetaCirculos = "/home/mateo/Escritorio/U/Vision_Computador/Unidad_3/Practicas/Practica_3.1/all-images/circle";
    string carpetaTriangulos = "/home/mateo/Escritorio/U/Vision_Computador/Unidad_3/Practicas/Practica_3.1/all-images/triangle";
    string carpetaCuadrados = "/home/mateo/Escritorio/U/Vision_Computador/Unidad_3/Practicas/Practica_3.1/all-images/square";

    // Calcular los momentos promedio para cada clase
    vector<double> momentosCirculo = calcularPromedioMomentos(carpetaCirculos, "Circulo");
    vector<double> momentosTriangulo = calcularPromedioMomentos(carpetaTriangulos, "Triangulo");
    vector<double> momentosCuadrado = calcularPromedioMomentos(carpetaCuadrados, "Cuadrado");

    // Normalizar los momentos de Hu
    momentosCirculo = normalizar(momentosCirculo);
    momentosTriangulo = normalizar(momentosTriangulo);
    momentosCuadrado = normalizar(momentosCuadrado);

    // Leer los momentos promedio desde el archivo CSV
    vector<pair<string, vector<double>>> momentosReferencia = leerMomentosDesdeCSV("momentos_hu.csv");

    // Cargar la imagen que se desea clasificar
    Mat img = imread("/home/mateo/Escritorio/U/Vision_Computador/Unidad_3/Practicas/Practica_3.1/preparacion/testing/c15i-1.PNG", IMREAD_COLOR);
    if (img.empty()) {
        cerr << "Error al cargar la imagen." << endl;
        return -1;
    }

    // Aplicar el mismo preprocesamiento a la imagen de prueba
    Mat imgPreprocesada = preprocesarImagen(img);

    // Calcular los momentos de Hu de la imagen preprocesada
    vector<double> momentosFigura = calcularMomentosHu(imgPreprocesada);

    // Clasificación por distancia
    double menorDistancia = DBL_MAX;
    string figuraClasificadaPorDistancia = "Desconocido";

    for (const auto& referencia : momentosReferencia) {
        double distancia = calcularDistancia(momentosFigura, referencia.second);
        cout << "Distancia a " << referencia.first << ": " << distancia << endl;
        if (distancia < menorDistancia) {
            menorDistancia = distancia;
            figuraClasificadaPorDistancia = referencia.first;
        }
    }

    cout << "Clasificación por distancia: " << figuraClasificadaPorDistancia << endl;

    // Mostrar las imágenes
    imshow("Imagen Original", img);
    imshow("Imagen Preprocesada", imgPreprocesada);

    waitKey(0);
    return 0;
}