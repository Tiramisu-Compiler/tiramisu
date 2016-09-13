#ifndef _H_DEBUG_
#define _H_DEBUG_

#include <iostream>

#define DEBUG 1
#define DEBUG2 0

#define IF_DEBUG2(x) {if (DEBUG2) {x;}};
#define IF_DEBUG(x)  {if (DEBUG || DEBUG2) {x;}};

namespace coli
{
	void str_dump(std::string str);
	void str_dump(std::string str, const char * str2);

	void error(std::string str, bool exit);
}

#endif
