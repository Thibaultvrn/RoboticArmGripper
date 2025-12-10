g++ -O3 -std=c++17 strawberry_classifier.cxx -o webcam_demo `pkg-config --cflags --libs opencv4`  \\
./webcam_demo          # viseur circulaire, détection uniquement rouge vif
