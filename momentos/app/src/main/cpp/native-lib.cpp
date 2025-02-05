#include <jni.h>
#include <string>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#include <cfloat>
#include <android/bitmap.h>
#include <android/log.h>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

#define LOG_TAG "native-lib"
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

// --------------------------------------------------------------------------
// Función: calcularDistancia
// Calcula la distancia Manhattan entre dos vectores.
double calcularDistancia(const vector<double>& a, const vector<double>& b) {
    double distancia = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        distancia += fabs(a[i] - b[i]);
    }
    return distancia;
}

// --------------------------------------------------------------------------
// Función: normalizar
// Normaliza un vector usando Z-score.
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
        // Evitar división por cero
        if (desviacion != 0)
            normalizado.push_back((v - media) / desviacion);
        else
            normalizado.push_back(0.0);
    }
    return normalizado;
}

// --------------------------------------------------------------------------
// Función: calcularMomentosHu
// Calcula los 7 momentos de Hu a partir de una imagen (se espera imagen ya preprocesada).
vector<double> calcularMomentosHu(const Mat& imagen) {
    Moments m = moments(imagen, true);
    double hu[7];
    HuMoments(m, hu);
    vector<double> momentos(hu, hu + 7);
    return momentos;
}

// --------------------------------------------------------------------------
// Función: transformarHu
// Aplica la transformación logarítmica a los momentos de Hu para mejorar su discriminación.
vector<double> transformarHu(const vector<double>& hu) {
    vector<double> huLog;
    for (double val : hu) {
        // Evitar log(0) y preservar el signo
        double trans = (fabs(val) > 1e-10) ? -copysign(log10(fabs(val)), val) : 0;
        huLog.push_back(trans);
    }
    return huLog;
}

// --------------------------------------------------------------------------
// Función: preprocesarImagen
// Modificada para utilizar operaciones morfológicas. Se convierte la imagen a escala de grises,
// se aplica un umbral inverso y se utiliza un "closing" morfológico para rellenar huecos.
// Finalmente se extrae y rellena el contorno de mayor área para generar una máscara sólida.
Mat preprocesarImagen(const Mat& img) {
    Mat gris, binary;

    // Convertir a escala de grises
    cvtColor(img, gris, COLOR_BGR2GRAY);

    // Aplicar umbral (similar a THRESH_BINARY_INV con valor 235)
    threshold(gris, binary, 235, 255, THRESH_BINARY_INV);

    // Operación morfológica: closing para rellenar huecos
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3,3));
    morphologyEx(binary, binary, MORPH_CLOSE, kernel);

    // Encontrar contornos en la imagen umbralizada
    vector<vector<Point>> contornos;
    vector<Vec4i> hierarchy;
    findContours(binary, contornos, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // Crear una máscara negra
    Mat mask = Mat::zeros(binary.size(), CV_8UC1);

    // Seleccionar el contorno con mayor área y rellenarlo en la máscara
    if (!contornos.empty()) {
        double maxArea = 0;
        int idxMax = -1;
        for (size_t i = 0; i < contornos.size(); i++) {
            double area = contourArea(contornos[i]);
            if (area > maxArea) {
                maxArea = area;
                idxMax = static_cast<int>(i);
            }
        }
        if (idxMax >= 0) {
            drawContours(mask, contornos, idxMax, Scalar(255), FILLED);
        }
    }
    return mask;
}

// --------------------------------------------------------------------------
// Función: leerMomentosDesdeCSV
// Lee el CSV de momentos (almacenado en assets) y retorna un vector de pares: (nombre_clase, vector_de_momentos).
vector<pair<string, vector<double>>> leerMomentosDesdeCSV(AAssetManager* mgr, const string& filename) {
    vector<pair<string, vector<double>>> momentos;
    AAsset* asset = AAssetManager_open(mgr, filename.c_str(), AASSET_MODE_STREAMING);
    if (!asset) {
        LOGE("No se pudo abrir el asset %s", filename.c_str());
        return momentos;
    }
    size_t fileLength = AAsset_getLength(asset);
    string fileContent;
    fileContent.resize(fileLength);
    AAsset_read(asset, &fileContent[0], fileLength);
    AAsset_close(asset);

    stringstream ss(fileContent);
    string linea;
    while(getline(ss, linea)) {
        stringstream ls(linea);
        string clase;
        getline(ls, clase, ',');
        vector<double> vec;
        string token;
        while(getline(ls, token, ',')) {
            try {
                vec.push_back(stod(token));
            } catch (...) {
                // Ignorar errores de conversión
            }
        }
        if (vec.size() == 7)
            momentos.push_back({clase, vec});
    }
    return momentos;
}

// --------------------------------------------------------------------------
// Función auxiliar: bitmapToMat
// Convierte un objeto Bitmap de Android a un cv::Mat (se asume formato ARGB_8888).
// Función auxiliar: bitmapToMat
// Convierte un objeto Bitmap de Android a un cv::Mat.
// Soporta tanto ANDROID_BITMAP_FORMAT_RGBA_8888 como ANDROID_BITMAP_FORMAT_RGB_565.
bool bitmapToMat(JNIEnv* env, jobject bitmap, Mat& mat) {
    AndroidBitmapInfo info;
    if (AndroidBitmap_getInfo(env, bitmap, &info) < 0) {
        LOGE("Error al obtener la información del Bitmap");
        return false;
    }

    void* pixels = nullptr;
    if (AndroidBitmap_lockPixels(env, bitmap, &pixels) < 0) {
        LOGE("Error al bloquear los píxeles del Bitmap");
        return false;
    }

    if (info.format == ANDROID_BITMAP_FORMAT_RGBA_8888) {
        // Procesamiento para formato RGBA_8888
        mat = Mat(info.height, info.width, CV_8UC4, pixels).clone();
        AndroidBitmap_unlockPixels(env, bitmap);
        cvtColor(mat, mat, COLOR_RGBA2BGR);
        return true;
    } else if (info.format == ANDROID_BITMAP_FORMAT_RGB_565) {
        // Procesamiento para formato RGB_565
        // Se creará una imagen de 3 canales (BGR)
        mat = Mat(info.height, info.width, CV_8UC3);
        uint16_t* src = (uint16_t*) pixels;
        for (int y = 0; y < info.height; y++) {
            for (int x = 0; x < info.width; x++) {
                uint16_t pixel = src[y * info.width + x];
                // Extraer cada canal:
                // - Rojo: bits 11-15 (5 bits)
                // - Verde: bits 5-10 (6 bits)
                // - Azul: bits 0-4 (5 bits)
                uint8_t r = (pixel >> 11) & 0x1F;
                uint8_t g = (pixel >> 5) & 0x3F;
                uint8_t b = pixel & 0x1F;
                // Escalar a 8 bits por canal
                r = (r * 255) / 31;
                g = (g * 255) / 63;
                b = (b * 255) / 31;
                mat.at<Vec3b>(y, x) = Vec3b(b, g, r);
            }
        }
        AndroidBitmap_unlockPixels(env, bitmap);
        return true;
    } else {
        LOGE("Formato de Bitmap no soportado: %d", info.format);
        AndroidBitmap_unlockPixels(env, bitmap);
        return false;
    }
}


// --------------------------------------------------------------------------
// Función nativa: procesarDibujo
// Se invoca desde MainActivity para procesar el dibujo realizado en el DrawView.
extern "C"
JNIEXPORT jstring JNICALL
Java_ec_edu_ups_momentos_MainActivity_procesarDibujo(JNIEnv *env, jobject /* this */, jobject bitmap, jobject assetManager) {
    // 1. Convertir el Bitmap a cv::Mat
    Mat imgOriginal;
    if (!bitmapToMat(env, bitmap, imgOriginal)) {
        return env->NewStringUTF("Error al convertir el Bitmap");
    }

    // 2. Preprocesar la imagen utilizando operaciones morfológicas
    //    Se obtiene una máscara sólida a partir del contorno de mayor área.
    Mat imgPreprocesada = preprocesarImagen(imgOriginal);

    // 3. Calcular los momentos de Hu y aplicar la transformación logarítmica
    vector<double> momentosFigura = calcularMomentosHu(imgPreprocesada);
    vector<double> momentosFiguraTrans = transformarHu(momentosFigura);
    vector<double> momentosFiguraNorm = normalizar(momentosFiguraTrans);

    // 4. Leer momentos almacenados en CSV para comparación
    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);
    vector<pair<string, vector<double>>> momentosBase = leerMomentosDesdeCSV(mgr, "momentos.csv");

    // 5. Clasificación por distancia mínima
    string mejorClase = "Desconocido";
    double menorDistancia = DBL_MAX;

    for (const auto& [clase, momentos] : momentosBase) {
        double distancia = calcularDistancia(momentosFiguraNorm, normalizar(momentos));
        if (distancia < menorDistancia) {
            menorDistancia = distancia;
            mejorClase = clase;
        }
    }

    // 6. Formatear los momentos para mostrarlos en pantalla
    string resultado = "Momentos de Hu:\n";
    for (size_t i = 0; i < momentosFiguraNorm.size(); i++) {
        resultado += "H" + to_string(i+1) + ": " + to_string(momentosFiguraNorm[i]) + "\n";
    }
    resultado += "\nClasificación: " + mejorClase;

    return env->NewStringUTF(resultado.c_str());
}
