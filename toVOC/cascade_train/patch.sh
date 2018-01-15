cd $HOME; mkdir .local
mkdir abs; cd abs
wget http://sourceforge.net/projects/opencvlibrary/files/opencv-unix/3.1.0/opencv-3.1.0.zip
unzip opencv-3.1.0.zip; cd opencv-3.1.0
cp ${RP_DIR}/scripts/cascade_train/opencv3.patch .
patch -p0 < opencv3.patch
mkdir build; cd build
cmake -D WITH_TBB=ON -D CMAKE_INSTALL_PREFIX=${HOME}/.local ..
make install -j8
