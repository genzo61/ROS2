cmake_minimum_required(VERSION 3.8)
project(my_robot_description)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
# Eğer ileride başka bağımlılık ekleyeceksen buraya ekle
# find_package(<dependency> REQUIRED)

# URDF dosyalarını install et
install(DIRECTORY urdf/
  DESTINATION share/${PROJECT_NAME}/urdf
)

# Mesh dosyalarını install et
install(DIRECTORY meshes/
  DESTINATION share/${PROJECT_NAME}/meshes
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # Linter copyright kontrolünü atlamak için
  set(ament_cmake_copyright_FOUND TRUE)
  # cpplint kontrolünü atlamak için
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
