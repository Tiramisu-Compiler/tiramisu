/*
 * Copyright (c) 2014, ARM Limited
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

/*
 * Header file to be included by programmer into pencil files.
 */

#ifndef _PENCIL_H
#define _PENCIL_H

/* The user is free to include these before pencil.h; We will need them anyway so include them in all cases and avoid different behaviour depending on the #include order. */
/* TODO: Except in pure mode */
#include <stdbool.h>/* bool */
#include <stdint.h> /* int32_t, int64_t */
#include <stdlib.h> /* strtod, abs, labs, llabs */
#include <math.h>   /* sin, sinf, ... */

/* Preprocessor aliases for PENCIL builtins.
 * They can be expanded here because the same magic happens using the __pencil_*
 * definitions.
 */
#define PENCIL_USE __pencil_use
#define PENCIL_DEF __pencil_def
#define PENCIL_MAYBE (__pencil_maybe())

/* For compatibility to the PENCIL spec. */
#define USE PENCIL_USE
#define DEF PENCIL_DEF
#define MAYBE PENCIL_MAYBE

#ifndef PRL_PENCIL_H
/* This is for outer C-code not itself PENCIL-code calling PENCIL functions. */
/* TODO: We maybe should put it into a separate header file because it is not
 * compatible to the PENCIL grammar. It is not sufficient to put it into the
 * "The file is processed as a C file" branch it would not be available to host
 * code when compiling embedded PENCIL functions using ppcg.
 */
enum npr_mem_tags {
	PENCIL_NPR_MEM_NOWRITE = 1,
	PENCIL_NPR_MEM_NOREAD = 2,
	PENCIL_NPR_MEM_NOACCESS = PENCIL_NPR_MEM_NOWRITE | PENCIL_NPR_MEM_NOREAD,
	PENCIL_NPR_MEM_READ = 4,
	PENCIL_NPR_MEM_WRITE = 8,
	PENCIL_NPR_MEM_READWRITE = PENCIL_NPR_MEM_READ | PENCIL_NPR_MEM_WRITE
};
void prl_npr_mem_tag(void *location, enum npr_mem_tags mode) __attribute__((weak));
static void __pencil_npr_mem_tag(void *location, enum npr_mem_tags mode) {
	if (&prl_npr_mem_tag)
		prl_npr_mem_tag(location, mode);
}
#endif /* PRL_PENCIL_H */

#ifdef __PENCIL__
/* The file is processed by the PENCIL-to-OpenCL code generator. */

/* Custom stdbool.h
 * We cannot directly include stdbool because its content might notconfiorm the
 * PENCIL grammar.
 */
#define bool _Bool
#define true 1
#define false 0
#define __bool_true_false_are_defined 1

/* PENCIL-specific macros */
#define ACCESS(...) PENCIL_ACCESS(__VA_ARGS__)

#define pencil_array static const restrict
#define PENCIL_ARRAY static const restrict

/* must define PENCIL_MATHDECL (either static or not) */
#include "pencil_mathdecl.h"

#else /* __PENCIL__ */
/* The file is processed as a C file. */

/* PENCIL functionality that require an #include in C */
#include <assert.h> /* assert */
#include <stdbool.h>/* bool */
#include <stdlib.h> /* abs, labs, llabs */
#include <math.h>   /* sqrt, sqrtf, ... */

#include "pencil_compat.h"
#include "pencil_mathimpl.h"

#endif /* __PENCIL__ */

#endif /* _PENCIL_H */
