##===----------------------------------------------------------------------===##
#
# Copyright (c) Intel Corporation
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# This file incorporates work covered by the following copyright and permission
# notice:
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
#
##===----------------------------------------------------------------------===##

# Installation path: <onedpl_root>/lib/cmake/oneDPL/
get_filename_component(_onedpl_root "${CMAKE_CURRENT_LIST_DIR}" REALPATH)
get_filename_component(_onedpl_root "${_onedpl_root}/../../../" ABSOLUTE)

if (WIN32)
    set(_onedpl_headers_subdir windows)
else()
    set(_onedpl_headers_subdir linux)
endif()


find_path(_onedpl_headers
  NAMES oneapi/dpl
  PATHS ${_onedpl_root}
  HINTS ENV DPL_ROOT_HINT
  PATH_SUFFIXES include ${_onedpl_headers_subdir}/include
)


if (EXISTS "${_onedpl_headers}")
    if (NOT TARGET oneDPL)
        include(CheckCXXCompilerFlag)

        add_library(oneDPL INTERFACE IMPORTED)
        set_target_properties(oneDPL PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${_onedpl_headers}")

        if (ONEDPL_PAR_BACKEND AND NOT ONEDPL_PAR_BACKEND MATCHES "^(tbb|openmp|serial)$")
            message(STATUS "oneDPL: ONEDPL_PAR_BACKEND=${ONEDPL_PAR_BACKEND} is requested, but not supported, available backends: tbb, openmp, serial")
            set(oneDPL_FOUND FALSE)
            return()
        endif()

        if (NOT ONEDPL_PAR_BACKEND OR ONEDPL_PAR_BACKEND STREQUAL "tbb")  # Handle oneTBB backend
            if (NOT TBB_FOUND)
               find_package(TBB 2021 QUIET COMPONENTS tbb PATHS ${CMAKE_SOURCE_DIR}/dpnp/backend/cmake/Modules NO_DEFAULT_PATH)
            endif()
            if (NOT TBB_FOUND AND ONEDPL_PAR_BACKEND STREQUAL "tbb")  # If oneTBB backend is requested explicitly, but not found.
                message(STATUS "oneDPL: ONEDPL_PAR_BACKEND=${ONEDPL_PAR_BACKEND} requested, but not found")
                set(oneDPL_FOUND FALSE)
                return()
            elseif (TBB_FOUND)
                set(ONEDPL_PAR_BACKEND tbb)
                message(STATUS "oneDPL: ONEDPL_PAR_BACKEND=${ONEDPL_PAR_BACKEND}, disable OpenMP backend")
                set_target_properties(oneDPL PROPERTIES INTERFACE_LINK_LIBRARIES TBB::tbb)
                set_property(TARGET oneDPL APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS ONEDPL_USE_TBB_BACKEND=1 ONEDPL_USE_OPENMP_BACKEND=0)
            endif()
        endif()

        if (NOT ONEDPL_PAR_BACKEND OR ONEDPL_PAR_BACKEND STREQUAL "openmp")  # Handle OpenMP backend
            if (UNIX)
                set(_openmp_flag "-fopenmp")
            else()
                set(_openmp_flag "-Qopenmp")
            endif()

            # Some compilers may fail if _openmp_flag is not in CMAKE_REQUIRED_LIBRARIES.
            set(_onedpl_saved_required_libs ${CMAKE_REQUIRED_LIBRARIES})
            set(CMAKE_REQUIRED_LIBRARIES ${_openmp_option})
            check_cxx_compiler_flag(${_openmp_flag} _openmp_option)
            set(CMAKE_REQUIRED_LIBRARIES ${_onedpl_saved_required_libs})
            unset(_onedpl_saved_required_libs)

            if (NOT _openmp_option AND ONEDPL_PAR_BACKEND STREQUAL "openmp")  # If OpenMP backend is requested explicitly, but not supported.
                message(STATUS "oneDPL: ONEDPL_PAR_BACKEND=${ONEDPL_PAR_BACKEND} requested, but not supported")
                set(oneDPL_FOUND FALSE)
                return()
            elseif (_openmp_option)
                set(ONEDPL_PAR_BACKEND openmp)
                message(STATUS "oneDPL: ONEDPL_PAR_BACKEND=${ONEDPL_PAR_BACKEND}, disable oneTBB backend")
                set_target_properties(oneDPL PROPERTIES INTERFACE_COMPILE_OPTIONS ${_openmp_flag})
                set_target_properties(oneDPL PROPERTIES INTERFACE_LINK_LIBRARIES ${_openmp_flag})
                set_property(TARGET oneDPL APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS ONEDPL_USE_TBB_BACKEND=0 ONEDPL_USE_OPENMP_BACKEND=1)
            endif()
        endif()

        if (NOT ONEDPL_PAR_BACKEND OR ONEDPL_PAR_BACKEND STREQUAL "serial")
            set(ONEDPL_PAR_BACKEND serial)
            message(STATUS "oneDPL: ONEDPL_PAR_BACKEND=${ONEDPL_PAR_BACKEND}, disable oneTBB and OpenMP backends")
            set_property(TARGET oneDPL APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS ONEDPL_USE_TBB_BACKEND=0 ONEDPL_USE_OPENMP_BACKEND=0)
        endif()

        check_cxx_compiler_flag("-fsycl" _fsycl_option)
        if (NOT _fsycl_option)
            message(STATUS "oneDPL: -fsycl is not supported by current compiler, set ONEDPL_USE_DPCPP_BACKEND=0")
            set_property(TARGET oneDPL APPEND PROPERTY INTERFACE_COMPILE_DEFINITIONS ONEDPL_USE_DPCPP_BACKEND=0)
        endif()
    endif()
else()
    message(STATUS "oneDPL: headers do not exist ${_onedpl_headers}")
    set(oneDPL_FOUND FALSE)
endif()
