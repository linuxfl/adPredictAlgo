/*************************************************************************
    > File Name: str_util.h
    > Author: starsnet83
    > Mail: starsnet83@gmail.com 
    > Created Time: 三 12/17 00:13:30 2014
 ************************************************************************/
#ifndef STR_UTIL_H
#define STR_UTIL_H

#include <string>
#include <vector>
#include <sstream>

namespace util
{
class str_util
{
	public:
		static void split(const std::string& line, const std::string delim, std::vector<std::string>& elems);

		static std::vector<std::string> split(const std::string& s, const std::string delim);

		static void trim(std::string& line);

		static void rtrim(std::string& line);

		static void ltrim(std::string& line);

		template<class T>
		T castFromS(const std::string& line)
		{
			T result;
			stream.clear();
			stream.str(line);
			stream >> result;
			stream.str("");
			return result;
		}

		template<class T>
		std::string castToS(const T& castT)
		{
			stream.clear();
			stream.str("");
			stream << castT;
			return stream.str();
		}

	private:
		std::stringstream stream;
};
}
#endif
