#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <filesystem>
#include <unordered_map>

using namespace cv;
using namespace std;
using namespace cv::ml;
namespace fs = std::filesystem;

// Mapa de las categorías (números) a los nombres de las categorías
unordered_map<int, string> categoryNames = {
    {1, "Batman"},
    {2, "Chrome"},
    {3, "Ebay"},
    {4, "Facebook"},
    {5, "Instagram"}
};

// Función para calcular el descriptor HOG con normalización
void computeHOG(Mat img, vector<float> &descriptors) {
    HOGDescriptor hog(
        Size(128, 128), 
        Size(16, 16),    
        Size(4, 4),     
        Size(8, 8),     
        18             
    );

    resize(img, img, Size(128, 128));
    GaussianBlur(img, img, Size(3, 3), 0);
    adaptiveThreshold(img, img, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2);
    equalizeHist(img, img);

    vector<Point> locations;
    hog.compute(img, descriptors, Size(8, 8), Size(0, 0), locations);

    // Normalizar las características HOG
    normalize(descriptors, descriptors, 0, 1, NORM_MINMAX);
}

// Función para obtener el bounding box dinámicamente
Rect getBoundingBoxForLogo(const Mat& img) {
    // Realizar la detección de bordes usando Canny (puedes usar otro método según tu necesidad)
    Mat edges;
    Canny(img, edges, 100, 200);

    // Encontrar los contornos en la imagen
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(edges, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Si se encuentran contornos, encontrar el bounding box del contorno más grande
    if (!contours.empty()) {
        double maxArea = 0;
        int maxIndex = -1;
        for (size_t i = 0; i < contours.size(); i++) {
            double area = contourArea(contours[i]);
            if (area > maxArea) {
                maxArea = area;
                maxIndex = i;
            }
        }

        if (maxIndex != -1) {
            // Obtener el bounding box del contorno más grande
            return boundingRect(contours[maxIndex]);
        }
    }

    // Si no se encuentra ningún contorno, devolver un bounding box vacío
    return Rect(0, 0, 0, 0);
}

// Función para realizar la predicción con el modelo SVM
string predictSVM(const Ptr<SVM>& svm, const vector<float>& descriptor, Rect& boundingBox, const Mat& img) {
    Mat testSample(1, descriptor.size(), CV_32F);
    for (size_t i = 0; i < descriptor.size(); i++) {
        testSample.at<float>(0, i) = descriptor[i];
    }

    // Obtener la respuesta del SVM (predicción)
    Mat response;
    svm->predict(testSample, response);

    // Obtener la distancia al hiperplano
    Mat dist;
    svm->predict(testSample, dist, StatModel::RAW_OUTPUT);

    // Calcula la confianza basándote en la distancia al hiperplano
    float confidence = dist.at<float>(0, 0);  // La distancia al hiperplano de decisión
    cout << "Confianza: " << confidence << endl;

    // Si la confianza es baja, consideramos que la clase es desconocida
    if (fabs(confidence) < 2) {  // Ajusta el umbral según sea necesario
        return "desconocido";
    }

    // De lo contrario, retornamos el nombre de la categoría
    int predictedClass = static_cast<int>(response.at<float>(0, 0));
    if (categoryNames.find(predictedClass) != categoryNames.end()) {
        // Aquí calculamos dinámicamente el bounding box basado en la región de interés detectada
        boundingBox = getBoundingBoxForLogo(img);  // Función para encontrar la región donde está el logo
        return categoryNames[predictedClass];
    }

    return "desconocido";
}


// Función para predecir un grupo de imágenes de prueba
void predictBatchSVM(const Ptr<SVM>& svm, const string& testFolderPath) {
    // Cargar las imágenes de test
    vector<Mat> testImages;
    vector<string> fileNames;
    for (const auto& entry : fs::directory_iterator(testFolderPath)) {
        Mat img = imread(entry.path().string(), IMREAD_GRAYSCALE);
        if (!img.empty()) {
            // Preprocesar la imagen antes de calcular el descriptor
            GaussianBlur(img, img, Size(3, 3), 0);
            adaptiveThreshold(img, img, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 11, 2);
            equalizeHist(img, img);

            testImages.push_back(img);
            fileNames.push_back(entry.path().filename().string());
        }
    }

    // Predicción para cada imagen
    for (size_t i = 0; i < testImages.size(); i++) {
        vector<float> descriptors;
        computeHOG(testImages[i], descriptors);  // Calcular el descriptor HOG para la imagen

        // Realizar la predicción
        Rect boundingBox;  // Variable para almacenar las coordenadas del bounding box
        string predictedLabel = predictSVM(svm, descriptors, boundingBox, testImages[i]);

        // Imprimir los resultados
        cout << "Imagen: " << fileNames[i] << " - Predicción: " << predictedLabel << endl;

        // Dibujar el cuadro delimitador (bounding box) en la imagen
        if (predictedLabel != "desconocido") {
            // Dibujar bounding box y etiqueta para logos conocidos
            rectangle(testImages[i], boundingBox, Scalar(0, 0, 255), 2);  // Rojo para conocidos
            putText(testImages[i], predictedLabel, Point(boundingBox.x, boundingBox.y - 10), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);  // Verde para conocidos
        } else {
            // Dibujar bounding box y etiqueta para "desconocido"
            rectangle(testImages[i], boundingBox, Scalar(255, 0, 0), 2);  // Azul para desconocido
            putText(testImages[i], "Desconocido", Point(boundingBox.x, boundingBox.y - 10), FONT_HERSHEY_SIMPLEX, 1, Scalar(255, 0, 0), 2);  // Azul para desconocido
        }

        // Mostrar la imagen con el bounding box y la etiqueta
        imshow("Predicción", testImages[i]);
        waitKey(0);  // Esperar por una tecla para continuar
    }
}




int main() {
    // Cargar el modelo SVM guardado
    Ptr<SVM> svm = SVM::load("logos_svm.xml");

    // Especificar la ruta de las imágenes de test
    string testFolderPath = "test";  // Cambia a la carpeta donde tienes las imágenes de test

    // Realizar la predicción sobre las imágenes de test
    predictBatchSVM(svm, testFolderPath);

    return 0;
}
