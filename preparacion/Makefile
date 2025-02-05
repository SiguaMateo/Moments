all:
	g++ -std=c++17 -lstdc++fs momentos_csv.cpp \
	-I//home/mateo/Aplicaciones/Librerias/opencv/opencvi/include/opencv4/ \
	-L//home/mateo/Aplicaciones/Librerias/opencv/opencvi/lib/ \
	-lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs \
	-lopencv_video -lopencv_videoio -lopencv_ml -lopencv_objdetect \
	-lopencv_features2d -lopencv_ximgproc -o vision.bin

run:
	./vision.bin
