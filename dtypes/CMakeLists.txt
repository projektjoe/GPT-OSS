cmake_minimum_required(VERSION 3.15)
project(gptoss_dtypes LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# bfloat16 bindings
pybind11_add_module(bfloat16 MODULE bfloat16.cpp)
target_include_directories(bfloat16 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Install the bfloat16 module
install(TARGETS bfloat16 LIBRARY DESTINATION dtypes COMPONENT python)


execute_process(
  COMMAND python3 -m site --user-site
  OUTPUT_VARIABLE PY_SITE
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(DNNL_ROOT "${PY_SITE}/onednn")

# Build linear_layer module without torch dependency
pybind11_add_module(linear_layer MODULE linear.cpp)
target_include_directories(linear_layer PRIVATE 
    ${DNNL_ROOT}/include
)

target_link_directories(linear_layer PRIVATE ${DNNL_ROOT})
target_link_libraries(linear_layer PRIVATE 
    dnnl 
    openblas
)

find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    target_link_libraries(linear_layer PRIVATE OpenMP::OpenMP_CXX)
endif()

# Install the linear_layer module
install(TARGETS linear_layer LIBRARY DESTINATION dtypes COMPONENT python)

# Conditionally build linear_layer_torch module if PyTorch is available
execute_process(
  COMMAND python3 -c "import torch; print(torch.utils.cmake_prefix_path)"
  OUTPUT_VARIABLE TORCH_PREFIX_PATH
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE TORCH_FIND_RESULT
)

if(TORCH_FIND_RESULT EQUAL 0)
  message(STATUS "PyTorch found. Building linear_layer_torch module.")
  set(CMAKE_PREFIX_PATH "${TORCH_PREFIX_PATH}")
  find_package(Torch REQUIRED)
  
  pybind11_add_module(linear_layer_torch MODULE linear_torch.cpp)
  target_include_directories(linear_layer_torch PRIVATE 
      ${DNNL_ROOT}/include
  )
  
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
  
  target_link_directories(linear_layer_torch PRIVATE ${DNNL_ROOT})
  target_link_libraries(linear_layer_torch PRIVATE 
      "${TORCH_LIBRARIES}"
  )
  
  if(OpenMP_CXX_FOUND)
      target_link_libraries(linear_layer_torch PRIVATE OpenMP::OpenMP_CXX)
  endif()
  
  # Set RPATH to find torch libraries at runtime
  set_target_properties(linear_layer_torch PROPERTIES
      INSTALL_RPATH_USE_LINK_PATH TRUE
      BUILD_WITH_INSTALL_RPATH FALSE
      INSTALL_RPATH "${TORCH_INSTALL_PREFIX}/lib"
  )
  
  # Get torch library directory for RPATH
  execute_process(
    COMMAND python3 -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"
    OUTPUT_VARIABLE TORCH_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  
  if(TORCH_LIB_DIR)
    set_target_properties(linear_layer_torch PROPERTIES
        INSTALL_RPATH "${TORCH_LIB_DIR}"
    )
    message(STATUS "Set RPATH for linear_layer_torch to: ${TORCH_LIB_DIR}")
  endif()
  
  # Install the linear_layer_torch module
  install(TARGETS linear_layer_torch LIBRARY DESTINATION dtypes COMPONENT python)
else()
  message(STATUS "PyTorch not found. Skipping linear_layer_torch module. To enable torch functionality, install PyTorch first: pip install torch")
endif()