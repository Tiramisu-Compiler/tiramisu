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
#define DEBUG 1

namespace coli
{
	void str_dump(std::string str);
	void str_dump(std::string str, const char * str2);

	void error(std::string str, bool exit);

	extern int coli_indentation;
}

/**
 * Run \p STMT if the debugging level is above \p LEVEL.
 */
#define IF_DEBUG(LEVEL,STMT) {if (DEBUG && DEBUG_LEVEL>=LEVEL) {\
								for (int coli_indent=0; coli_indent<coli::coli_indentation; coli_indent++)\
									coli::str_dump(" ");\
								STMT;}};

/**
 * Change the indentation printed before running IF_DEBUG.
 * Useful to indent the text printed by IF_DEBUG.
 */
#define DEBUG_INDENT(x) {coli_indentation = coli_indentation + x;}

#endif
