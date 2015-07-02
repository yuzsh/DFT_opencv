# 下記の2つは自分の環境に合わせて書き換える
SET(OpenCV_DIR "/usr/local/Cellar/opencv/2.4.11_1/bin")
# OpenCVConfig.cmake があるディレクトリ
INCLUDE_DIRECTORIES("/usr/local/Cellar/opencv/2.4.11_1/include")
# 上記ディレクトリの下のincludeディレクトリ

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT( X_OpenCV )
FIND_PACKAGE( OpenCV REQUIRED )
ADD_EXECUTABLE( TEST 
DFT.cpp
 ) #任意のcppプログラム
TARGET_LINK_LIBRARIES( TEST ${OpenCV_LIBS})