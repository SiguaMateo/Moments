#include <iostream>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include <opencv2/ximgproc.hpp> // Para la esqueletización
#include <fstream>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

// Función para calcular los Momentos de Hu después de la esqueletización
void calculateHuMoments(const Mat &image, vector<double> &huMoments)
{
    Size targetSize(160, 160); // Tamaño objetivo
    Mat resizedImage;
    resize(image, resizedImage, targetSize, 0, 0, INTER_LINEAR);

    // Convertir la imagen a escala de grises
    Mat gray;
    cvtColor(resizedImage, gray, COLOR_BGR2GRAY);
    // imshow("Original", image);

    // Aplicar umbral binario con inversión de colores
    Mat binary;
    threshold(gray, binary, 235, 255, THRESH_BINARY_INV);
    // imshow("Threshold Inverted", binary);

    // Definir el kernel para la erosión y dilatación
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));

    // Aplicar erosión para eliminar pequeños ruidos
    Mat eroded;
    erode(binary, eroded, kernel, Point(-1, -1), 1);
    // imshow("Eroded", eroded);

    // Aplicar dilatación para restaurar la estructura de la figura
    Mat dilated;
    dilate(eroded, dilated, kernel, Point(-1, -1), 1);
    // imshow("Dilated", dilated);

    // Esqueletización usando el método de Zhang-Suen
    Mat skeleton;
    ximgproc::thinning(dilated, skeleton, ximgproc::THINNING_ZHANGSUEN);
    // imshow("Skeleton", skeleton);

    // Calcular los momentos a partir de la imagen esqueletizada
    Moments m = moments(skeleton, true);

    // Calcular los momentos de Hu
    double hu[7];
    HuMoments(m, hu);

    // Guardar los valores en el vector
    huMoments.assign(hu, hu + 7);

    // waitKey(0);
    // destroyAllWindows();
}

int main()
{
    string basePath = "/home/mateo/Escritorio/U/Vision_Computador/Unidad_3/Practicas/Practica_3.1/all-images";   // Ruta base del dataset
    string outputCSV = "figureshu.csv"; // Archivo de salida
    ofstream datasetFile(outputCSV);

    // Recorrer cada carpeta (circle, triangle, square)
    for (const auto &dirEntry : fs::directory_iterator(basePath))
    {
        if (fs::is_directory(dirEntry))
        {
            string className = dirEntry.path().filename().string(); // Nombre de la clase (circle, triangle, square)
            for (const auto &fileEntry : fs::directory_iterator(dirEntry.path()))
            {
                if (fs::is_regular_file(fileEntry))
                {
                    string filePath = fileEntry.path().string();            // Ruta completa de la imagen
                    string fileName = fileEntry.path().filename().string(); // Nombre del archivo

                    // Leer la imagen
                    Mat image = imread(filePath);

                    // Verificar si la imagen fue leída correctamente
                    if (image.empty())
                    {
                        cerr << "Error al leer la imagen: " << filePath << endl;
                        continue;
                    }

                    // Calcular los 7 Momentos de Hu
                    vector<double> huMoments;
                    calculateHuMoments(image, huMoments);

                    // Escribir los datos en el archivo CSV
                    datasetFile << className << "," << fileName;
                    for (const auto &moment : huMoments)
                    {
                        datasetFile << "," << moment;
                    }
                    datasetFile << "\n";
                }
            }
        }
    }

    // Cerrar el archivo CSV
    datasetFile.close();

    cout << "Dataset creado exitosamente en " << outputCSV << endl;
    return 0;
}
