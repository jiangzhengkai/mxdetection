
#include "c_api.hpp"
#include "permutohedral_c.hpp"

int test(int n)
{
	return n + 3;
}

int PermutohedralCreate(PermutohedralHandle *out)
{
	*out = new Permutohedral();
	return 0;
}

int PermutohedralFree(PermutohedralHandle handle)
{
	delete static_cast<Permutohedral*>(handle);
	return 0;
}

int PermutohedralInit(PermutohedralHandle handle, const float* features, int fea_size, int N)
{
	Permutohedral *permutohedral = static_cast<Permutohedral*>(handle);
	permutohedral->init(features, fea_size, N);
	return 0;
}

int PermutohedralCompute(PermutohedralHandle handle, float* out, const float* in, int in_size)
{
	Permutohedral *permutohedral = static_cast<Permutohedral*>(handle);
	permutohedral->compute(out, in, in_size);
	return 0;
}

