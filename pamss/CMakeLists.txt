cmake_minimum_required(VERSION 2.6)

# SETUP A MAKE CLEAN TO CLEAN EVERYTHING
SET_DIRECTORY_PROPERTIES( PROPERTIES ADDITIONAL_MAKE_CLEAN_FILES 
      "CMakeCache.txt;cmake_install.cmake" )

# DEFAULT BUILD: RELEASE
if (NOT CMAKE_BUILD_TYPE)
   message(STATUS "No build type selected, default to Release")
   set(CMAKE_BUILD_TYPE "Release")
   # set(CMAKE_BUILD_TYPE "Debug")
endif()


# IIO
if( NOT TARGET IIOLIB )
   INCLUDE_DIRECTORIES("iio")
   ADD_SUBDIRECTORY(iio)
ENDIF() 
SET( LIBS ${LIBS} IIOLIB )


# FFTW3 
#SET( LIBS "${LIBS}" fftw3 m )

## OpenMP
#FIND_PACKAGE( OpenMP )
#IF(OPENMP_FOUND)
#   MESSAGE("OPENMP FOUND")
##   set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
##   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#   SET( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_C_FLAGS}" )
#ENDIF()


# ADD LOCAL PATH TO SEARCH MODULES
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
# eigen
FIND_PACKAGE(Eigen3)
INCLUDE_DIRECTORIES( ${EIGEN3_INCLUDE_DIR} )


## IF THE COMPILER IS NOT GCC FAIL!
#if(CMAKE_COMPILER_IS_GNUCC)
#   SET( EXTRA_COMPILER_FLAGS "${EXTRA_COMPILER_FLAGS} -std=c99 -Wall -Wextra -pedantic" )
#else()
#   MESSAGE("This project needs c99")
#endif()


# Enable C99
if (CMAKE_VERSION VERSION_LESS "3.1")
   set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")
else ()
   set (CMAKE_C_STANDARD 99)
endif ()

# Enable C++11
if (CMAKE_VERSION VERSION_LESS "3.1")
   set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11") # -fsanitize=address)
else ()
   set (CMAKE_CXX_STANDARD 11)
endif ()
SET( EXTRA_COMPILER_FLAGS "${EXTRA_COMPILER_FLAGS} ") #  -fsanitize=address -Wall -Wextra -pedantic" )

# set the cnpy library path
set(CNPY_PATH "${PROJECT_SOURCE_DIR}/cnpy")
include_directories("${CNPY_PATH}/build/include")
link_directories("${CNPY_PATH}/build")

ADD_EXECUTABLE( pamss main.cpp img.cpp img.hpp imgio.cpp rag.cpp rag.hpp model.hpp mergetreelog.cpp mergetreelog.h)
TARGET_LINK_LIBRARIES( pamss ${LIBS} cnpy z )
SET_TARGET_PROPERTIES( pamss PROPERTIES COMPILE_FLAGS "${EXTRA_COMPILER_FLAGS} ${OpenMP_C_FLAGS}" )

ADD_EXECUTABLE( mstree treecut.cpp img.cpp img.hpp imgio.cpp rag.cpp rag.hpp model.hpp mergetreelog.cpp mergetreelog.h)
TARGET_LINK_LIBRARIES( mstree ${LIBS} cnpy z )
SET_TARGET_PROPERTIES( mstree PROPERTIES COMPILE_FLAGS "${EXTRA_COMPILER_FLAGS} ${OpenMP_C_FLAGS}" )

