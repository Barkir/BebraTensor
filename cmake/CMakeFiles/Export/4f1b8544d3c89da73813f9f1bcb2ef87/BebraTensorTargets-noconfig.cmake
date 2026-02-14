#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Bebra::bebra_core" for configuration ""
set_property(TARGET Bebra::bebra_core APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(Bebra::bebra_core PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_NOCONFIG "CXX"
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib64/libbebra_core.a"
  )

list(APPEND _cmake_import_check_targets Bebra::bebra_core )
list(APPEND _cmake_import_check_files_for_Bebra::bebra_core "${_IMPORT_PREFIX}/lib64/libbebra_core.a" )

# Import target "Bebra::bebra_tensor" for configuration ""
set_property(TARGET Bebra::bebra_tensor APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(Bebra::bebra_tensor PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/bin/bebra_tensor"
  )

list(APPEND _cmake_import_check_targets Bebra::bebra_tensor )
list(APPEND _cmake_import_check_files_for_Bebra::bebra_tensor "${_IMPORT_PREFIX}/bin/bebra_tensor" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
