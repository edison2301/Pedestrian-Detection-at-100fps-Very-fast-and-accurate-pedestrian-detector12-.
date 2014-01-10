# This is a CMake build file, for more information consult:
# http://en.wikipedia.org/wiki/CMake
# and
# http://www.cmake.org/Wiki/CMake
# http://www.cmake.org/cmake/help/syntax.html
# http://www.cmake.org/Wiki/CMake_Useful_Variables
# http://www.cmake.org/cmake/help/cmake-2-8-docs.html

# to compile the local code you can use: cmake ./ && make -j2

# ----------------------------------------------------------------------
# Base CMake setup

cmake_minimum_required (VERSION 2.6)

set(biclop_root "../../..")

set(CMAKE_MODULE_PATH $ENV{CMAKE_MODULE_PATH})
set(CMAKE_MODULE_PATH "./" ${biclop_root} ${CMAKE_MODULE_PATH})
#set(CMAKE_MODULE_PATH "/home/rodrigob/work/code/biclop_references/cuda/FindCUDA/CMake/cuda" ${CMAKE_MODULE_PATH})
set(CMAKE_MODULE_PATH "/users/visics/mmathias/no_backup/code/references/cuda/FindCUDA/CMake/cuda" ${CMAKE_MODULE_PATH})

# ----------------------------------------------------------------------
# Setup the project

include(FindPkgConfig)
project (ObjectsDetection)

# ----------------------------------------------------------------------
# Site specific configurations
include(${biclop_root}/common_settings.cmake)

# ----------------------------------------------------------------------
# Setup required libraries
pkg_check_modules(libpng REQUIRED libpng)
#pkg_check_modules(OpenEXR REQUIRED OpenEXR)
#pkg_check_modules(opencv REQUIRED opencv>=2.3)
#set(vw_LIBRARIES "-lvwCore -lvwImage -lvwStereo -lvwFileIO -lvwMath -lvwInterestPoint")

set(opencv_LIBRARIES
    opencv_core opencv_imgproc opencv_highgui opencv_ml
    opencv_video opencv_features2d
    opencv_calib3d
    #opencv_objdetect opencv_contrib
    opencv_legacy opencv_flann
   ) # quick hack for opencv2.4 support

# ----------------------------------------------------------------------
# Setup CUDA
if(USE_GPU)
  find_package(CUDA 4.0 REQUIRED)
  include_directories(${CUDA_INCLUDE_DIRS} ${CUDA_CUT_INCLUDE_DIR})
endif(USE_GPU)

# ----------------------------------------------------------------------
# Setup link and include directories

set(local_LIBRARY_DIRS
  "/usr/local/lib"
  "/users/visics/mmathias/no_backup/usr/local/lib"
  "/usr/lib64"
  "/usr/lib64/atlas"
  "/usr/lib/sse2/atlas"
  "/usr/lib/llvm-2.8/lib"
  ${local_CUDA_LIB_DIR}
)
set(local_INCLUDE_DIRS
  "/users/visics/mmathias/no_backup/usr/local/include"
  "/usr/include/eigen2/"
   "/usr/local/include/eigen2"
   "/usr/local/cuda-5.5/include"
   ${CUDA_INCLUDE_DIRS}
)

link_directories(
  ${libpng_LIBRARY_DIRS}
  ${OpenEXR_LIBRARY_DIRS}
  ${opencv_LIBRARY_DIRS}
  ${local_LIBRARY_DIRS}
)

include_directories(
  "${biclop_root}/libs"
  "${biclop_root}/src"
  ${libpng_INCLUDE_DIRS}
  ${OpenEXR_INCLUDE_DIRS}
  ${opencv_INCLUDE_DIRS}
  ${local_INCLUDE_DIRS}
  "${biclop_root}/libs/cudatemplates/include"
)

if(USE_GPU)
cuda_include_directories("${biclop_root}/libs/")
endif(USE_GPU)
        
# ----------------------------------------------------------------------
# Collect source files

set(biclop_src "${biclop_root}/src")
set(biclop_stereo "${biclop_root}/src/stereo_matching")

file(GLOB SrcCpp
  "./ObjectsDetection*.cpp"
  "./draw*.cpp"
  "${biclop_src}/*.cpp"
  #"${biclop_src}/objects_detection/*.c*"
  "${biclop_src}/objects_detection/Abstract*.c*"
  "${biclop_src}/objects_detection/*Converter.c*"
  "${biclop_src}/objects_detection/Base*.c*"
  "${biclop_src}/objects_detection/*Factory.c*"
  "${biclop_src}/objects_detection/Greedy*.c*"
  "${biclop_src}/objects_detection/Detection*.c*"
  "${biclop_src}/objects_detection/*Model.c*"
  "${biclop_src}/objects_detection/*Stage.c*"
  "${biclop_src}/objects_detection/*Integral*.c*"
  "${biclop_src}/objects_detection/SearchRange*.c*"
  "${biclop_src}/objects_detection/MultiscalesIntegral*.c*"
  "${biclop_src}/objects_detection/integral_channels/Integral*.cpp"
  "${biclop_src}/objects_detection/FastestPedestrian*.c*"
  "${biclop_src}/objects_detection/DetectorSearchRange.c*"
  "${biclop_src}/objects_detection/*.pb.c*"
  "${biclop_src}/objects_detection/cascade_stages/*.c*"
  "${biclop_src}/objects_detection/non_maximal_suppression/*.c*"

  "${biclop_src}/objects_tracking/*.cpp"
  "${biclop_src}/objects_tracking/motion_models/*.cpp"
  "${biclop_src}/objects_tracking/detection_descriptors/*.cpp"
  "${biclop_src}/objects_tracking/tracked_detections/*.cpp"

  "${biclop_src}/applications/*.cpp"
  "${biclop_src}/applications/stixel_world/*Gui.cpp"
  "${biclop_src}/applications/objects_tracking/*Application.cpp"
  "${biclop_src}/visual_odometry/Abstract*.cpp"

  #"${biclop_stereo}/*.cpp"
  "${biclop_stereo}/cost_volume/*CostVolume.cpp"
  "${biclop_stereo}/cost_volume/*CostVolumeEstimator*.cpp"
  "${biclop_stereo}/cost_volume/DisparityCostVolumeFromDepthMap.cpp"
  "${biclop_stereo}/cost_functions.cpp"
  "${biclop_stereo}/CensusCostFunction.cpp"
  "${biclop_stereo}/CensusTransform.cpp"
  "${biclop_stereo}/GradientTransform.cpp"
  "${biclop_stereo}/AbstractStereoMatcher.cpp"
  "${biclop_stereo}/AbstractStereoBlockMatcher.cpp"
  "${biclop_stereo}/SimpleBlockMatcher.cpp"
  "${biclop_stereo}/MutualInformationCostFunction.cpp"
  "${biclop_stereo}/ConstantSpaceBeliefPropagation.cpp"
  "${biclop_stereo}/qingxiong_yang/*.cpp"
  "${biclop_stereo}/SimpleTreesOptimizationStereo.cpp"
  "${biclop_stereo}/OpenCvStereo.cpp"

  "${biclop_stereo}/ground_plane/*.cpp"
  "${biclop_stereo}/stixels/*.cpp"
   #"${biclop_stereo}/stixels/*.cc"
  "${biclop_src}/video_input/*.cpp"
  "${biclop_src}/video_input/calibration/*.c*"
  "${biclop_src}/video_input/preprocessing/*.cpp"
  #"${biclop_src}/features_tracking/*.cpp"
  "${biclop_src}/image_processing/*.cpp"

  "${biclop_src}/drawing/gil/*.cpp"
)


file(GLOB HelpersCpp
  #"${biclop_src}/helpers/*.cpp"
  "${biclop_src}/helpers/data/*.c*"
  "${biclop_src}/helpers/any_to_string.cpp"
  "${biclop_src}/helpers/get_section_options.cpp"
  "${biclop_src}/helpers/Log.cpp"
  "${biclop_src}/helpers/loggers.cpp"
  "${biclop_src}/helpers/AlignedImage.cpp"
  "${biclop_src}/helpers/replace_environment_variables.cpp"
  "${biclop_src}/helpers/Timers.cpp"
  "${biclop_src}/helpers/objects_detection/*.cpp"
)

file(GLOB SrcGpuCpp
  "${biclop_src}/objects_detection/Gpu*.cpp"
  "${biclop_src}/objects_detection/integral_channels/Gpu*.cpp"
  "${biclop_src}/helpers/gpu/*.cpp"

  #"${biclop_stereo}/SimpleTreesGpuStereo.cpp"
)

file(GLOB SrcCuda
  "${biclop_src}/objects_detection/integral_channels/gpu/*.cu"
  "${biclop_src}/objects_detection/integral_channels/gpu/*.cpp"
  "${biclop_src}/objects_detection/gpu/*.cu"
  "${biclop_src}/objects_detection/gpu/*.cpp"

  #"${biclop_src}/helpers/gpu/*.cu"

#  "${biclop_stereo}/*.cu.c*"
#  "${biclop_stereo}/*.cu"
#  "${biclop_stereo}/gpu/*.cu.c*"
#  "${biclop_stereo}/gpu/*.cu"
)

list(REMOVE_ITEM SrcCpp ${SrcCuda}) # just in case


if(USE_GPU)

# add GPU related source code to the executable list
list(APPEND SrcCpp ${SrcGpuCpp})

# add GPU related libraries
list(APPEND opencv_LIBRARIES opencv_gpu)

# ----------------------------------------------------------------------
# Compile CUDA stuff
cuda_include_directories(${local_CUDA_CUT_INCLUDE_DIRS})
cuda_include_directories(${CUDA_INCLUDE_DIRS} ${CUDA_CUT_INCLUDE_DIR} ${local_CUDA_CUT_INCLUDE_DIR})
link_directories(${local_CUDA_CUT_LIBRARY_DIRS})

cuda_add_library(cuda_stuff_library ${SrcCuda})
target_link_libraries(cuda_stuff_library
   ${CUDA_LIBRARIES}
   ${cutil_LIB}
)

#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --generate-line-info) # used during profiling
  
endif(USE_GPU)
# ----------------------------------------------------------------------
# Create the executable
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x") # required for unrestricted unions
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -p") # add gprof information

add_library(cpp_stuff_library  ${SrcCpp} ${HelpersCpp})

add_executable(objects_detection "./objects_detection.cpp")

target_link_libraries(objects_detection

   cpp_stuff_library

   ${cg_LIBRARIES}
   # linking with CgGL _after_ boost_program_options generates a segmentation fault ! boost_program_options 1.39 has a bug

   boost_program_options-mt boost_filesystem-mt boost_system-mt boost_thread-mt
   protobuf pthread
   SDL X11 Xext #Xrandr
   gomp
   ${libpng_LIBRARIES} jpeg
#  ${OpenEXR_LIBRARIES} 
   ${opencv_LIBRARIES}

   #${vw_LIBRARIES}
   #csparse sparse spblas mv
   #lapack blas atlas

   #${google_perftools_LIBS} # enables profiling, see http://code.google.com/p/gperftools

   #`OcelotConfig -l`
   #ocelot
   #boost_system-mt boost_filesystem-mt boost_thread-mt
   #GLEW
   #LLVMAsmParser LLVMX86Disassembler LLVMX86AsmParser LLVMX86CodeGen LLVMSelectionDAG
   #LLVMAsmPrinter LLVMMCParser LLVMX86AsmPrinter LLVMX86Info LLVMJIT
   #LLVMExecutionEngine LLVMCodeGen LLVMScalarOpts LLVMInstCombine LLVMTransformUtils LLVMipa
   #LLVMAnalysis LLVMTarget LLVMMC LLVMCore LLVMSupport LLVMSystem
)   

if(USE_GPU)
target_link_libraries(objects_detection cuda_stuff_library ${local_CUDA_LIB})
endif(USE_GPU)
# ----------------------------------------------------------------------
