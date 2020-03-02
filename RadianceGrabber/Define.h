#include <d3d11.h>
#pragma warning( disable : 4819 )
#include <curand_kernel.h>
#include <cassert>
#pragma warning( disable : 4819 )
#include <cuda_runtime.h>

#pragma once

namespace RadGrabber
{
#ifdef __CUDACC__
#define KLAUNCH_ARGS2(func, grid, block) func <<< grid, block >>>
#define KLAUNCH_ARGS3(func, grid, block, sh_mem) func <<< grid, block, sh_mem >>>
#define KLAUNCH_ARGS4(func, grid, block, sh_mem, stream) func <<< grid, block, sh_mem, stream >>>
#else
#define KLAUNCH_ARGS2(func, grid, block) func
#define KLAUNCH_ARGS3(func, grid, block, sh_mem) func
#define KLAUNCH_ARGS4(func, grid, block, sh_mem, stream) func
#endif

#define SAFE_HOST_DELETE(x) if(x) delete x
#define SAFE_HOST_DELETE_ARRAY(x) if (x) delete[] x
#define SAFE_DEVICE_DELETE(x) if(x) cudaFree(x)

#define SAFE_HOST_FREE(x) if(x) free(x)

#define ASSERT(x) assert(!x)
#define ASSERT_IS_NOT_NULL(x) assert(x != nullptr)
#define ASSERT_IS_NULL(x) assert(x == nullptr)
#define ASSERT_IS_FALSE(x) assert(!x)
#define ASSERT_IS_TRUE(x) assert(x)

#define OUT 
#define IN 
#define INOUT 
#define OUT_BUF 
#define IN_BUF 
#define DEL 

	typedef char byte;

	typedef char int8;
	typedef unsigned char uint8;

	typedef short int16;
	typedef unsigned short uint16;

	typedef int int32;
	typedef unsigned int uint32;
}