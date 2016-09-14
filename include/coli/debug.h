#ifndef _H_DEBUG_
#define _H_DEBUG_

#include <iostream>

#define DEBUG_LEVEL 2
#define DEBUG 1

#define IF_DEBUG(LEVEL,x) {if (DEBUG && DEBUG_LEVEL>=LEVEL) {x;}};

namespace coli
{
	void str_dump(std::string str);
	void str_dump(std::string str, const char * str2);

	void error(std::string str, bool exit);
}

#endif
