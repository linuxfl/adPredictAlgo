/*************************************************************************
    > File Name: sparse_fea.h
    > Author: starsnet83
    > Mail: starsnet83@gmail.com 
    > Created Time: 四 12/25 00:09:45 2014
 ************************************************************************/

#ifndef SPARSE_FEA_H
#define SPARSE_FEA_H

#include <vector>

namespace fea
{
struct fea_item
{
	uint32_t fea_index;
};

typedef std::vector<uint32_t> fea_list;

struct instance
{
	int label;
	fea_list fea_vec;
	void reset()
	{
		label = 0;
		fea_vec.clear();
	}
};
}
#endif
