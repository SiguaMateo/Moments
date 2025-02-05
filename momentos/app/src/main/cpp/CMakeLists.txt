# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html.
# For more examples on how to use CMake, see https://github.com/android/ndk-samples.

# Sets the minimum CMake version required for this project.
cmake_minimum_required(VERSION 3.22.1)

# Declares the project name. The project name can be accessed via ${ PROJECT_NAME},
# Since this is the top level CMakeLists.txt, the project name is also accessible
# with ${CMAKE_PROJECT_NAME} (both CMake variables are in-sync within the top level
# build script scope).
project("momentos")

set(OpenCV_STATIC on)
set(OpenCV_DIR /home/mateo/Aplicaciones/Librerias/opencv/OpenCV-android-sdk/sdk/native/jni)
find_package(OpenCV REQUIRED)
include_directories(/home/mateo/Aplicaciones/Librerias/opencv/OpenCV-android-sdk/sdk/native/jni/
        include)

# Añadir librería nativa
add_library(native-lib SHARED native-lib.cpp)

find_library(log-lib log)
find_library(android-lib android)

target_link_libraries(native-lib ${log-lib} android jnigraphics opencv_core opencv_highgui opencv_imgcodecs opencv_imgproc opencv_video opencv_videoio opencv_objdetect)

target_link_libraries(native-lib
        ${log-lib}
        ${android-lib}
        ${OpenCV_LIBS}
)