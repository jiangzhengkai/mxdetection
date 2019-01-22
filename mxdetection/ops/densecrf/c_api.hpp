
#ifndef _PERMUTOHEDRAL_C_API_H
#define _PERMUTOHEDRAL_C_API_H

/*! \brief Inhibit C++ name-mangling for MXNet functions. */
#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus


/*! \brief DLL prefix for windows */
#ifdef _WIN32
#ifdef NEWONE_EXPORTS
#define NEWONE_DLL __declspec(dllexport)
#else
#define NEWONE_DLL __declspec(dllimport)
#endif
#else
#define NEWONE_DLL
#endif

	typedef void *PermutohedralHandle;

	NEWONE_DLL int test(int n);

	NEWONE_DLL int PermutohedralCreate(PermutohedralHandle *out);

	NEWONE_DLL int PermutohedralFree(PermutohedralHandle handle);

	NEWONE_DLL int PermutohedralInit(PermutohedralHandle handle, const float* features, int fea_size, int N);

	NEWONE_DLL int PermutohedralCompute(PermutohedralHandle handle, float* out, const float* in, int in_size);


#ifdef __cplusplus
}
#endif  // __cplusplus

#endif  // _PERMUTOHEDRAL_C_API_H