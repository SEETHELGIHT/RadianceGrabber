#pragma warning( disable : 4819 )
#include <d3d11.h>
#include <curand_kernel.h>
#include <cassert>
#include <cuda_runtime.h>
#define _CRT_SECURE_NO_WARNINGS
#include <cstdarg>
#include <cstdio>
#include <device_launch_parameters.h>

#pragma once

namespace RadGrabber
{
#ifdef __CUDACC__
#define KLAUNCH_ARGS2(func, grid, block, ...) func <<< grid, block >>>(__VA_ARGS__)
#define KLAUNCH_ARGS3(func, grid, block, sh_mem, ...) func <<< grid, block, sh_mem >>>(__VA_ARGS__)
#define KLAUNCH_ARGS4(func, grid, block, sh_mem, stream, ...) func <<< grid, block, sh_mem, stream >>>(__VA_ARGS__)
#else
#define KLAUNCH_ARGS2(func, grid, block, ...) 
#define KLAUNCH_ARGS3(func, grid, block, sh_mem, ...) 
#define KLAUNCH_ARGS4(func, grid, block, sh_mem, stream, ...) 
#endif

#define SAFE_HOST_DELETE(x) if(x) delete x, x = nullptr
#define SAFE_HOST_DELETE_ARRAY(x) if (x) delete[] x, x = nullptr
#define SAFE_DEVICE_DELETE(x) if(x) cudaFree(x)

#define SAFE_HOST_FREE(x) if(x) { free(x); (x) = nullptr; }

#define ASSERT(x) assert((x))
#define ASSERT_IS_NOT_NULL(x) assert(x != nullptr)
#define ASSERT_IS_NULL(x) assert(x == nullptr)
#define ASSERT_IS_FALSE(x) assert(!(x))
#define ASSERT_IS_TRUE(x) assert(x)

#define OUT 
#define IN 
#define INOUT 
#define OUT_BUF 
#define IN_BUF 
#define DEL 

	typedef unsigned char byte;

	typedef char int8;
	typedef unsigned char uint8;

	typedef short int16;
	typedef unsigned short uint16;

	typedef int int32;
	typedef unsigned int uint32;

	__forceinline__ __device__
		int getGlobalIdx_1D_1D() {
		return blockIdx.x *blockDim.x + threadIdx.x;
	}

	__forceinline__ __device__
		int getGlobalIdx_1D_2D() {
		return blockIdx.x * blockDim.x * blockDim.y
			+ threadIdx.y * blockDim.x + threadIdx.x;
	}

	__forceinline__ __device__
		int getGlobalIdx_1D_3D() {
		return blockIdx.x * blockDim.x * blockDim.y * blockDim.z
			+ threadIdx.z * blockDim.y * blockDim.x
			+ threadIdx.y * blockDim.x + threadIdx.x;
	}
	__forceinline__ __device__ int getGlobalIdx_2D_1D() {
		int blockId = blockIdx.y * gridDim.x + blockIdx.x;
		int threadId = blockId * blockDim.x + threadIdx.x;
		return threadId;
	}
	__forceinline__ __device__
		int getGlobalIdx_2D_2D() {
		int blockId = blockIdx.x + blockIdx.y * gridDim.x;
		int threadId = blockId * (blockDim.x * blockDim.y)
			+ (threadIdx.y * blockDim.x) + threadIdx.x;
		return threadId;
	}
	__forceinline__ __device__
		int getGlobalIdx_2D_3D() {
		int blockId = blockIdx.x + blockIdx.y * gridDim.x;
		int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
			+ (threadIdx.z * (blockDim.x * blockDim.y))
			+ (threadIdx.y * blockDim.x) + threadIdx.x;
		return threadId;
	}
	__forceinline__ __device__
		int getGlobalIdx_3D_1D() {
		int blockId = blockIdx.x + blockIdx.y * gridDim.x
			+ gridDim.x * gridDim.y * blockIdx.z;
		int threadId = blockId * blockDim.x + threadIdx.x;
		return threadId;
	}
	__forceinline__ __device__
		int getGlobalIdx_3D_2D() {
		int blockId = blockIdx.x + blockIdx.y * gridDim.x
			+ gridDim.x * gridDim.y * blockIdx.z;
		int threadId = blockId * (blockDim.x * blockDim.y)
			+ (threadIdx.y * blockDim.x) + threadIdx.x;
		return threadId;
	}
	__forceinline__ __device__
		int getGlobalIdx_3D_3D() {
		int blockId = blockIdx.x + blockIdx.y * gridDim.x
			+ gridDim.x * gridDim.y * blockIdx.z;
		int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
			+ (threadIdx.z * (blockDim.x * blockDim.y))
			+ (threadIdx.y * blockDim.x) + threadIdx.x;
		return threadId;
	}
}