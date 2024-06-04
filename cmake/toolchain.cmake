if(NOT TOOLCHAIN)
  set(TOOLCHAIN "x86_64-linux-gcc10" CACHE STRING "Build toolchain." FORCE)
endif()
message(STATUS "Toolchain: ${TOOLCHAIN}")

include("./cmake/toolchains/${TOOLCHAIN}.cmake")
