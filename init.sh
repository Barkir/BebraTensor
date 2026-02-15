mkdir cmake
cd cmake

echo "
@PACKAGE_INIT@

include(CMakeFindDependencyMacro)
find_dependency(Protobuf REQUIRED)

include("${CMAKE_CURRENT_LIST_DIR}/BebraTensorTargets.cmake")

check_required_components(BebraTensor)
" >> BebraTensorConfig.cmake.in

cmake ..
cmake --build .
ctest
