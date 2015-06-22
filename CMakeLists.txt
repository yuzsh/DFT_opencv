# 下記の2つは自分の環境に合わせて書き換える
SET(OpenCV_DIR "C:\bin_OpenCV") # OpenCVConfig.cmake があるディレクトリ
INCLUDE_DIRECTORIES("C:\bin_OpenCV\include") # 上記ディレクトリの下のincludeディレクトリ

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)
PROJECT( SampleOpenCV )
FIND_PACKAGE( OpenCV REQUIRED )
ADD_EXECUTABLE( MyProject 
DFT.cpp
 ) #任意のcppプログラム
TARGET_LINK_LIBRARIES( MyProject ${OpenCV_LIBS})