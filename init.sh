mkdir cmake
cd cmake

echo "
@PACKAGE_INIT@

include(CMakeFindDependencyMacro)
find_dependency(Protobuf REQUIRED)

include("${CMAKE_CURRENT_LIST_DIR}/BebraTensorTargets.cmake")

check_required_components(BebraTensor)
" >> BebraTensorConfig.cmake.in

pip3 install questionary

cmake ..
cmake --build .
ctest
