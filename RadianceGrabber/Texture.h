#pragma once

#include "BinaryLayout.h"
#include "host_defines.h"

namespace RadGrabber
{
	class Texture
	{
	public:
		__host__ Texture();
		__host__ Texture(void* p, int pixelSize, int pixelCount);
		__host__ Texture(const Texture& t);
		__host__ Texture(const Texture&& t);
		__host__ Texture(const TextureChunk* c);
		__host__ ~Texture();

	protected:
		__host__ bool CopyFromDevice(const Texture& m);
		__host__ bool BuildFromChunk(const TextureChunk* p);

	private:
		__host__ __device__ int pixelCount;
		__host__ __device__ int pixelItemSize;
		__host__ __device__ int pixelItemDim;
		__host__ __device__ void* mTexturePtr;
	};
}
