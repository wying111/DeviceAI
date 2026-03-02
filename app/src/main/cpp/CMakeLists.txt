cmake_minimum_required(VERSION 3.22.1) # 你的 CMake 版本

project("ai_infra_core") # 假设你的项目名

# 1. 引入 PyTorch C++ 头文件搜索路径
# 这样 #include <torch/script.h> 就能被找到了
include_directories(${PYTORCH_DIR}/headers)

# 2. 导入 PyTorch 的预编译动态库 (libpytorch_jni_lite.so)
add_library(pytorch_jni_lite SHARED IMPORTED)
set_target_properties(pytorch_jni_lite PROPERTIES
        IMPORTED_LOCATION ${PYTORCH_DIR}/jni/${ANDROID_ABI}/libpytorch_jni_lite.so
)

# 找到 Android 自带的库
find_library(log-lib log)
find_library(jnigraphics-lib jnigraphics) # Bitmap 处理必须的库

# 3. 声明你自己的 C++ 库 (根据你报错的信息，你的库名叫 native-lib 或者跟项目同名)
add_library(
        ${CMAKE_PROJECT_NAME}
        SHARED
        native-lib.cpp)

# 4. 链接所有库
target_link_libraries(
        ${CMAKE_PROJECT_NAME}
        ${log-lib}
        ${jnigraphics-lib}
        pytorch_jni_lite # 链接 PyTorch 动态库
)