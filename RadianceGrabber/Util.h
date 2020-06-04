#include <memory>
#include "Define.h"

#pragma once

namespace RadGrabber
{
#ifndef REMOVE_LOG
	void Log(const char* format, ...);
	void __declspec(dllexport) __stdcall SetBlockLog(int isBlock);
	void SetFilePtr(FILE* ptr);
	void __declspec(dllexport) __stdcall FlushLog();
#else
#define Log(x, ...) 
#define SetBlockLog(b) 
#define SetFilePtr(p) 
#define FlushLog() 
#endif

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

	inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
	{
		if (code != cudaSuccess)
		{
			Log("Assertion: %s %s %d\n", cudaGetErrorString(code), file, line);
			if (abort) assert(code);
		}
	}

	inline void* MAllocDevice(int cb)
	{
		void* ptr = nullptr;
		gpuErrchk(cudaMalloc(&ptr, cb));
		return ptr;
	}

	inline void* MAllocManaged(int cb)
	{
		void* ptr = nullptr;
		gpuErrchk(cudaMallocManaged(&ptr, cb));
		return ptr;
	}

	inline void* MAllocHost(int cb, int flags = 0U)
	{
		return malloc(cb);
		void* ptr = nullptr;
		gpuErrchk(cudaMallocHost(&ptr, cb, flags));
		return ptr;
	}

	template<typename T>
	inline T* MAllocHost(int count, int flags = 0U)
	{
		T* ptr = nullptr;
		gpuErrchk(cudaMallocHost(&ptr, count * sizeof(T), flags));
		return ptr;
	}

	template<typename T>
	inline void MCopy(void* dst, const void* src, int count, cudaMemcpyKind kind)
	{
		gpuErrchk(cudaMemcpy(dst, src, count * sizeof(T), kind));
	}

	inline void MCopy(void* dst, const void* src, int size, cudaMemcpyKind kind)
	{
		gpuErrchk(cudaMemcpy(dst, src, size, kind));
	}

	inline void MCopyToSymbol(void* dst, const void* src, int size)
	{
		gpuErrchk(cudaMemcpyToSymbol(dst, src, size));
	}

	inline void MSet(void* dst, int value, size_t count)
	{
		gpuErrchk(cudaMemset(dst, value, count));
	}
}