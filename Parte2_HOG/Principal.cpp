#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <iostream>
#include <vector>
#include <filesystem>
#include <fstream>
#include <random>

using namespace cv;
using namespace std;
using namespace cv::ml;
namespace fs = std::filesystem;

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

// Función para aumentar el dataset con más rotaciones y escalados
void augmentImage(const Mat &img, vector<Mat> &augmentedImages, int classLabel) {
    augmentedImages.push_back(img.clone());

    // Rotaciones (-20° a 20°)
    for (int angle = -20; angle <= 20; angle += 5) {
        Mat rotated;
        Point2f center(img.cols / 2.0, img.rows / 2.0);
        Mat rotationMatrix = getRotationMatrix2D(center, angle, 1.0);
        warpAffine(img, rotated, rotationMatrix, img.size());
        augmentedImages.push_back(rotated);
    }

    // Escalado (80%-120%)
    for (double scale = 0.8; scale <= 1.2; scale += 0.2) {
        Mat scaled;
        resize(img, scaled, Size(), scale, scale);
        augmentedImages.push_back(scaled);
    }

    // Reflejo horizontal
    Mat flipped;
    flip(img, flipped, 1);
    augmentedImages.push_back(flipped);
}

// Función para cargar imágenes y extraer características HOG
void loadDataset(const string &path, vector<Mat> &images, vector<int> &labels, int classLabel, const string &outputPath) {
    for (const auto &entry : fs::directory_iterator(path)) {
        Mat img = imread(entry.path().string(), IMREAD_GRAYSCALE);
        if (!img.empty()) {
            vector<Mat> augmentedImages;
            augmentImage(img, augmentedImages, classLabel);

            int imgCounter = 0;
            for (const auto &augImg : augmentedImages) {
                string savePath = outputPath + "/" + to_string(classLabel) + "_" + to_string(imgCounter++) + ".png";
                imwrite(savePath, augImg);
                images.push_back(augImg);
                labels.push_back(classLabel);
            }
        }
    }
}

// Función para entrenar el clasificador SVM
void trainSVM(vector<Mat> &images, vector<int> &labels) {
    vector<vector<float>> trainingData;
    for (auto &img : images) {
        vector<float> descriptors;
        computeHOG(img, descriptors);
        trainingData.push_back(descriptors);
    }

    Mat trainData(trainingData.size(), trainingData[0].size(), CV_32F);
    for (size_t i = 0; i < trainingData.size(); i++) {
        for (size_t j = 0; j < trainingData[i].size(); j++) {
            trainData.at<float>(i, j) = trainingData[i][j];
        }
    }

    Mat trainLabels(labels.size(), 1, CV_32S, labels.data());

    Ptr<SVM> svm = SVM::create();
    svm->setType(SVM::C_SVC);
    svm->setKernel(SVM::LINEAR);  // Cambiar a un kernel no lineal
    svm->setC(10.0);  // Ajusta C según sea necesario
    svm->setGamma(0.5);  // Ajusta el parámetro gamma para el kernel RBF
    svm->setTermCriteria(TermCriteria(TermCriteria::MAX_ITER, 5000, 1e-7));

    cout << "Entrenando el modelo SVM con parámetros optimizados y HOG..." << endl;
    svm->train(trainData, ROW_SAMPLE, trainLabels);
    cout << "Entrenamiento completado." << endl;

    svm->save("logos_svm.xml");
    cout << "Modelo guardado en 'logos_svm.xml'." << endl;
}

// Función para predicción con el modelo SVM
string predictSVM(const Ptr<SVM>& svm, const vector<float>& descriptor) {
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
    if (fabs(confidence) < 0.5) {  // Ajusta el umbral según sea necesario
        return "desconocido";
    }

    // De lo contrario, retornamos la clase predicha
    return response.at<float>(0, 0) == -1 ? "desconocido" : to_string(static_cast<int>(response.at<float>(0, 0)));
}

// Función principal
int main() {
    vector<Mat> images;
    vector<int> labels;
    string outputPath = "dataset_augmented";

    fs::create_directories(outputPath);

    // Cargar datasets de diferentes clases
    loadDataset("images/batman", images, labels, 1, outputPath);
    loadDataset("images/chrome", images, labels, 2, outputPath);
    loadDataset("images/ebay", images, labels, 3, outputPath);
    loadDataset("images/facebook", images, labels, 4, outputPath);
    loadDataset("images/instagram", images, labels, 5, outputPath);

    cout << "Total de imágenes tras aumentación: " << images.size() << endl;

    // Dividir en conjunto de entrenamiento y prueba (80% entrenamiento, 20% prueba)
    int trainSize = static_cast<int>(images.size() * 0.8);  // 80% para entrenamiento
    vector<Mat> trainImages(images.begin(), images.begin() + trainSize);
    vector<int> trainLabels(labels.begin(), labels.begin() + trainSize);

    vector<Mat> testImages(images.begin() + trainSize, images.end());
    vector<int> testLabels(labels.begin() + trainSize, labels.end());

    // Entrenamiento del modelo SVM con el conjunto de entrenamiento
    trainSVM(trainImages, trainLabels);

    // Cargar el modelo SVM guardado
    Ptr<SVM> svm = SVM::load("logos_svm.xml");

    // Predicción en el conjunto de prueba
    int correct = 0;
    for (size_t i = 0; i < testImages.size(); i++) {
        vector<float> descriptors;
        computeHOG(testImages[i], descriptors);

        // Predicción
        string predictedLabel = predictSVM(svm, descriptors);

        if (predictedLabel == to_string(testLabels[i])) {
            correct++;
        }
    }

    // Mostrar el porcentaje de aciertos
    float accuracy = static_cast<float>(correct) / testImages.size() * 100.0;
    cout << "Precisión del modelo en el conjunto de prueba: " << accuracy << "%" << endl;

    return 0;
}
