#include <memory>
#include "Define.h"

#pragma once

namespace RadGrabber
{
	inline void* MAllocDevice(int cb)
	{
		void* ptr = nullptr;
		ASSERT_IS_FALSE(cudaMalloc(&ptr, cb));
		return ptr;
	}

	inline void* MAllocHost(int cb, int flags = 0U)
	{
		void* ptr = nullptr;
		ASSERT_IS_FALSE(cudaMallocHost(&ptr, cb, flags));
		return ptr;
	}

	inline void MCopy(void* dst, const void* src, int size, cudaMemcpyKind kind)
	{
		ASSERT_IS_FALSE(cudaMemcpy(dst, src, size, kind));
	}

	inline void MCopyToSymbol(void* dst, const void* src, int size, int offset, cudaMemcpyKind kind)
	{
		ASSERT_IS_FALSE(cudaMemcpyToSymbol(dst, src, size, offset, kind));
	}

	inline void MSet(void* dst, int value, size_t count)
	{
		ASSERT_IS_FALSE(cudaMemset(dst, 0, count));
	}

	template<typename T>
	inline T* MAllocHost(int count, int flags = 0U)
	{
		T* ptr = nullptr;
		ASSERT_IS_FALSE(cudaMAllocHost(&ptr, count * sizeof(T), flags));
		return ptr;
	}
}