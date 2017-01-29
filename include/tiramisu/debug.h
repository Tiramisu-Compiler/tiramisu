#ifndef _H_DEBUG_
#define _H_DEBUG_

#include <iostream>

/**
 * Debugging level.
 */
#define DEBUG_LEVEL 1

/**
 * Set to 1 to enable debugging and 0 to disable debugging.
 */
#define ENABLE_DEBUG 1

namespace tiramisu
{
    void str_dump(std::string str);
    void str_dump(std::string str, const char * str2);
    void str_dump(const char * str, const char * str2);
    void print_indentation();

    void error(std::string str, bool exit);

    extern int tiramisu_indentation;
}

/**
 * Print function name.
 */
#define DEBUG_FCT_NAME(LEVEL) {if (ENABLE_DEBUG && DEBUG_LEVEL>=LEVEL) {\
                                tiramisu::print_indentation();\
                                tiramisu::str_dump("[");\
                                tiramisu::str_dump(__FUNCTION__);\
                                tiramisu::str_dump(" function]\n");}}

/**
 * Run \p STMT if the debugging level is above \p LEVEL.
 */
#define DEBUG(LEVEL,STMT) {if (ENABLE_DEBUG && DEBUG_LEVEL>=LEVEL) {\
                                tiramisu::print_indentation();\
                                STMT;\
                                tiramisu::str_dump("\n");}};

/**
 * Run \p STMT if the debugging level is above \p LEVEL.
 * If \p NEW_LINE is set to true, then a new line is printed at
 * the end of DEBUG.
 */
#define DEBUG_NO_NEWLINE(LEVEL,STMT) {if (ENABLE_DEBUG && DEBUG_LEVEL>=LEVEL) {\
                                tiramisu::print_indentation();\
                                STMT;}};

/**
 * Change the indentation printed before running IF_DEBUG.
 * Useful to indent the text printed by IF_DEBUG.
 */
#define DEBUG_INDENT(x) {tiramisu::tiramisu_indentation = tiramisu::tiramisu_indentation + x;}

#endif
