## CMakeLists.txt
## Summary: Build system for ipa — island parser schema compiler and runtime library.
##
## Author:  KaisarCode
## Website: https://kaisarcode.com
## License: https://www.gnu.org/licenses/gpl-3.0.html

cmake_minimum_required(VERSION 3.14)
project(ipa C CXX)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

option(IPA_USE_VENDORED_LIBS "Build ipa against vendored lib snapshots" ON)

if(NOT IPA_USE_VENDORED_LIBS)
    message(FATAL_ERROR "External prebuilt kclib artifacts are not supported by default")
endif()

## ── Compiler flags ────────────────────────────────────────────────────────────

if(NOT MSVC)
    add_compile_options(-O3 -fno-finite-math-only -Wall -Wextra -Werror)
    add_definitions(-D_GNU_SOURCE -DGGML_USE_CPU -DGGML_VERSION="1" -DGGML_COMMIT="custom")
else()
    add_compile_options(/O2 /W3 /WX)
    add_definitions(-DGGML_USE_CPU -DGGML_VERSION="1" -DGGML_COMMIT="custom")
endif()

## ── Target platform ───────────────────────────────────────────────────────────

if(NOT DEFINED KC_PLATFORM OR KC_PLATFORM STREQUAL "")
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set(KC_PLATFORM "linux")
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        set(KC_PLATFORM "windows")
    elseif(ANDROID)
        set(KC_PLATFORM "android")
    else()
        set(KC_PLATFORM "linux")
    endif()
endif()

## ── Vendored dependency sources ───────────────────────────────────────────────

set(IPA_VENDOR_INCLUDES
    src
    lib
    lib/ggml/include
    lib/ggml/src
    lib/ggml/src/ggml-cpu
)

set(IPA_KCLIB_SOURCES
    lib/libemb.c
    lib/libhnsw.c
    lib/libmmap.c
    lib/libngram.c
)

file(GLOB GGML_BASE_SOURCES
    "lib/ggml/src/*.c"
    "lib/ggml/src/*.cpp"
    "lib/ggml/src/ggml-cpu/*.c"
    "lib/ggml/src/ggml-cpu/*.cpp"
)

set(GGML_ARCH_SOURCES "")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    file(GLOB GGML_ARCH_SOURCES
        "lib/ggml/src/ggml-cpu/arch/x86/*.c"
        "lib/ggml/src/ggml-cpu/arch/x86/*.cpp"
    )
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "i686|i386")
    file(GLOB GGML_ARCH_SOURCES
        "lib/ggml/src/ggml-cpu/arch/x86/*.c"
        "lib/ggml/src/ggml-cpu/arch/x86/*.cpp"
    )
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|armv7|arm")
    file(GLOB GGML_ARCH_SOURCES
        "lib/ggml/src/ggml-cpu/arch/arm/*.c"
        "lib/ggml/src/ggml-cpu/arch/arm/*.cpp"
    )
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64")
    file(GLOB GGML_ARCH_SOURCES
        "lib/ggml/src/ggml-cpu/arch/riscv/*.c"
        "lib/ggml/src/ggml-cpu/arch/riscv/*.cpp"
    )
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "powerpc64le|ppc64le")
    file(GLOB GGML_ARCH_SOURCES
        "lib/ggml/src/ggml-cpu/arch/powerpc/*.c"
    )
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "mips")
    set(GGML_ARCH_SOURCES "")
    add_definitions(-DGGML_CPU_GENERIC)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "s390x")
    file(GLOB GGML_ARCH_SOURCES
        "lib/ggml/src/ggml-cpu/arch/s390/*.c"
        "lib/ggml/src/ggml-cpu/arch/s390/*.cpp"
    )
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "loongarch64")
    file(GLOB GGML_ARCH_SOURCES
        "lib/ggml/src/ggml-cpu/arch/loongarch/*.c"
    )
endif()

set(GGML_SOURCES ${GGML_BASE_SOURCES} ${GGML_ARCH_SOURCES})
set(GGML_C_SOURCES "")
set(GGML_CXX_SOURCES "")

foreach(GGML_SOURCE ${GGML_SOURCES})
    get_filename_component(GGML_SOURCE_EXT "${GGML_SOURCE}" EXT)
    if(GGML_SOURCE_EXT STREQUAL ".c")
        list(APPEND GGML_C_SOURCES "${GGML_SOURCE}")
    elseif(GGML_SOURCE_EXT STREQUAL ".cpp")
        list(APPEND GGML_CXX_SOURCES "${GGML_SOURCE}")
    endif()
endforeach()

if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(
        ${GGML_C_SOURCES}
        PROPERTIES COMPILE_OPTIONS "-Wno-pointer-to-int-cast;-Wno-psabi;-Wno-unused-function;-Wno-unused-variable"
    )
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set_source_files_properties(
        ${GGML_CXX_SOURCES}
        PROPERTIES COMPILE_OPTIONS "-Wno-psabi;-Wno-unused-function;-Wno-unused-variable"
    )
endif()

## ── Embedded model ────────────────────────────────────────────────────────────

set(MODEL_GGUF "${CMAKE_SOURCE_DIR}/lib/model.gguf")

if(NOT EXISTS "${MODEL_GGUF}")
    message(FATAL_ERROR "lib/model.gguf is required to compile vendored libemb.c")
endif()

if(CMAKE_SYSTEM_NAME STREQUAL "Windows")
    set(MODEL_OBJ     "${CMAKE_BINARY_DIR}/model.o")
    set(MODEL_OBJ_ELF "${CMAKE_BINARY_DIR}/model_elf.o")

    find_program(OBJCOPY_HOST NAMES objcopy)
    get_filename_component(COMPILER_DIR "${CMAKE_C_COMPILER}" DIRECTORY)
    get_filename_component(COMPILER_NAME "${CMAKE_C_COMPILER}" NAME_WE)
    string(REPLACE "gcc" "objcopy" OBJCOPY_CROSS_NAME "${COMPILER_NAME}")
    find_program(OBJCOPY_CROSS
        NAMES "${OBJCOPY_CROSS_NAME}" objcopy
        HINTS "${COMPILER_DIR}"
        PATHS "${COMPILER_DIR}"
    )

    if(NOT OBJCOPY_HOST OR NOT OBJCOPY_CROSS)
        message(FATAL_ERROR "objcopy not found (host=${OBJCOPY_HOST} cross=${OBJCOPY_CROSS})")
    endif()

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "i686|i386")
        set(PE_FORMAT "pe-i386")
        set(ELF_INTERMEDIATE "elf32-i386")
    else()
        set(PE_FORMAT "pe-x86-64")
        set(ELF_INTERMEDIATE "elf64-x86-64")
    endif()

    add_custom_command(
        OUTPUT "${MODEL_OBJ}"
        COMMAND "${OBJCOPY_HOST}"
                --input-target binary --output-target "${ELF_INTERMEDIATE}"
                model.gguf "${MODEL_OBJ_ELF}"
        COMMAND "${OBJCOPY_CROSS}"
                --input-target "${ELF_INTERMEDIATE}" --output-target "${PE_FORMAT}"
                "${MODEL_OBJ_ELF}" "${MODEL_OBJ}"
        DEPENDS "${MODEL_GGUF}"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/lib"
        COMMENT "Embedding lib/model.gguf into model.o"
    )
else()
    set(MODEL_OBJ "${CMAKE_BINARY_DIR}/model.o")

    if(ANDROID)
        find_program(OBJCOPY_CMD NAMES llvm-objcopy objcopy)
    else()
        get_filename_component(COMPILER_DIR "${CMAKE_C_COMPILER}" DIRECTORY)
        get_filename_component(COMPILER_NAME "${CMAKE_C_COMPILER}" NAME_WE)
        if("${COMPILER_NAME}" STREQUAL "cc")
            set(OBJCOPY_NAME "objcopy")
        else()
            string(REPLACE "gcc" "objcopy" OBJCOPY_NAME "${COMPILER_NAME}")
        endif()
        find_program(OBJCOPY_CMD
            NAMES "${OBJCOPY_NAME}" objcopy
            HINTS "${COMPILER_DIR}"
            PATHS "${COMPILER_DIR}"
        )
    endif()

    if(NOT OBJCOPY_CMD)
        message(FATAL_ERROR "objcopy not found for this toolchain")
    endif()

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        set(OBJCOPY_FORMAT "elf64-littleaarch64")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "armv7|arm")
        set(OBJCOPY_FORMAT "elf32-littlearm")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64")
        set(OBJCOPY_FORMAT "elf64-littleriscv")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "powerpc64le|ppc64le")
        set(OBJCOPY_FORMAT "elf64-powerpcle")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "mips64")
        set(OBJCOPY_FORMAT "elf64-tradlittlemips")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "mipsel")
        set(OBJCOPY_FORMAT "elf32-tradlittlemips")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "mips")
        set(OBJCOPY_FORMAT "elf32-tradbigmips")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "s390x")
        set(OBJCOPY_FORMAT "elf64-s390")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "loongarch64")
        set(OBJCOPY_FORMAT "elf64-loongarch")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "i686|i386")
        set(OBJCOPY_FORMAT "elf32-i386")
    else()
        set(OBJCOPY_FORMAT "elf64-x86-64")
    endif()

    add_custom_command(
        OUTPUT "${MODEL_OBJ}"
        COMMAND "${OBJCOPY_CMD}"
                --input-target binary
                --output-target "${OBJCOPY_FORMAT}"
                model.gguf "${MODEL_OBJ}"
        DEPENDS "${MODEL_GGUF}"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}/lib"
        COMMENT "Embedding lib/model.gguf into model.o"
    )
endif()

set_source_files_properties("${MODEL_OBJ}" PROPERTIES GENERATED TRUE EXTERNAL_OBJECT TRUE)
set(IPA_VENDOR_SOURCES ${IPA_KCLIB_SOURCES} ${GGML_SOURCES} "${MODEL_OBJ}")

## ── Static library ────────────────────────────────────────────────────────────

add_library(ipa_static STATIC src/libipa.c ${IPA_VENDOR_SOURCES})

set_target_properties(ipa_static PROPERTIES OUTPUT_NAME "ipa" POSITION_INDEPENDENT_CODE ON)

target_include_directories(ipa_static PRIVATE ${IPA_VENDOR_INCLUDES})

## ── Shared library ────────────────────────────────────────────────────────────

add_library(ipa_shared SHARED src/libipa.c ${IPA_VENDOR_SOURCES})

set_target_properties(ipa_shared PROPERTIES OUTPUT_NAME "ipa")

target_include_directories(ipa_shared PRIVATE ${IPA_VENDOR_INCLUDES})

if(UNIX AND NOT ANDROID)
    target_link_libraries(ipa_shared pthread dl m stdc++)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "mips")
        target_link_options(ipa_shared PRIVATE -Wl,--no-warn-mismatch)
    endif()
elseif(ANDROID)
    target_link_libraries(ipa_shared dl m stdc++)
elseif(WIN32)
    target_link_libraries(ipa_shared stdc++)
endif()

## ── Executable ────────────────────────────────────────────────────────────────

add_executable(ipa src/ipa.c src/libipa.c ${IPA_VENDOR_SOURCES})

target_include_directories(ipa PRIVATE ${IPA_VENDOR_INCLUDES})

if(UNIX AND NOT ANDROID)
    target_link_libraries(ipa pthread dl m stdc++)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "armv7|arm|mips")
        target_link_libraries(ipa atomic)
    endif()
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "mips")
        target_link_options(ipa PRIVATE -Wl,--no-warn-mismatch)
    endif()
elseif(ANDROID)
    target_link_libraries(ipa dl m stdc++)
elseif(WIN32)
    target_link_libraries(ipa stdc++)
endif()
