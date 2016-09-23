#ifndef _H_DEBUG_
#define _H_DEBUG_

#include <iostream>

/**
 * Debugging level.
 */
#define DEBUG_LEVEL 2

/**
 * Set to 1 to enable debugging and 0 to disable debugging.
 */
#define ENABLE_DEBUG 1

namespace coli
{
    void str_dump(std::string str);
    void str_dump(std::string str, const char * str2);
    void str_dump(const char * str, const char * str2);
    void print_indentation();

    void error(std::string str, bool exit);

    extern int coli_indentation;
}

/**
 * Print function name.
 */
#define DEBUG_FCT_NAME(LEVEL) {if (ENABLE_DEBUG && DEBUG_LEVEL>=LEVEL) {\
                                coli::print_indentation();\
                                coli::str_dump("[");\
                                coli::str_dump(__FUNCTION__);\
                                coli::str_dump(" function]:\n");}}

/**
 * Print function name.
 */
#define DEBUG_FCT_NAME_END(LEVEL) {if (ENABLE_DEBUG && DEBUG_LEVEL>=LEVEL) {\
                                coli::print_indentation();\
                                coli::str_dump("[");\
                                coli::str_dump(__FUNCTION__);\
                                coli::str_dump(" function end].\n");}}

/**
 * Run \p STMT if the debugging level is above \p LEVEL.
 */
#define DEBUG(LEVEL,STMT) {if (ENABLE_DEBUG && DEBUG_LEVEL>=LEVEL) {\
                                coli::print_indentation();\
                                STMT;\
                                coli::str_dump("\n");}};

/**
 * Run \p STMT if the debugging level is above \p LEVEL.
 * If \p NEW_LINE is set to true, then a new line is printed at
 * the end of DEBUG.
 */
#define DEBUG_NO_NEWLINE(LEVEL,STMT) {if (ENABLE_DEBUG && DEBUG_LEVEL>=LEVEL) {\
                                coli::print_indentation();\
                                STMT;}};

/**
 * Change the indentation printed before running IF_DEBUG.
 * Useful to indent the text printed by IF_DEBUG.
 */
#define DEBUG_INDENT(x) {coli::coli_indentation = coli::coli_indentation + x;}

#endif
