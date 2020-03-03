#include "ColorTarget.h"
#include "MemUtil.h"
#include "Unity/RenderAPI.h"

namespace RadGrabber
{
	__host__ SingleFrameColorTarget::SingleFrameColorTarget(Vector2i textureResolution, void * texturePtr) : mTextureResolution(textureResolution), mTexturePtr(texturePtr)
	{ }

	__host__ SingleFrameColorTarget::~SingleFrameColorTarget()
	{ }

	__host__ void SingleFrameColorTarget::UpdateColorFromHost(int colorCount, const ColorRGB * colorPtr, int startIndex)
	{
		RenderAPI* api = RenderAPI::GetRenderAPI();
		ColorRGB* mappedPtr = reinterpret_cast<ColorRGB*>(api->BeginWriteTexture2D(mTexturePtr));
		MCopy(mappedPtr + startIndex, colorPtr, colorCount * sizeof(ColorRGB), cudaMemcpyKind::cudaMemcpyHostToHost);
		api->EndWriteTexture2D(mTexturePtr);
	}

	__host__ void SingleFrameColorTarget::UpdateColorFromDevice(int colorCount, const ColorRGB * deviceColorPtr, int startIndex)
	{
		RenderAPI* api = RenderAPI::GetRenderAPI();
		ColorRGB* mappedPtr = reinterpret_cast<ColorRGB*>(api->BeginWriteTexture2D(mTexturePtr));
		MCopy(mappedPtr + startIndex, deviceColorPtr, colorCount * sizeof(ColorRGB), cudaMemcpyKind::cudaMemcpyDeviceToHost);
		api->EndWriteTexture2D(mTexturePtr);
	}

	__host__ void SingleFrameColorTarget::CopyColorToHost(int startIndex, int colorCount, ColorRGB * colorPtr) const
	{
		RenderAPI* api = RenderAPI::GetRenderAPI();
		ColorRGB* mappedPtr = reinterpret_cast<ColorRGB*>(api->BeginReadTexture2D(mTexturePtr));
		MCopy(colorPtr, mappedPtr + startIndex, colorCount * sizeof(ColorRGB), cudaMemcpyKind::cudaMemcpyHostToHost);
		api->EndReadTexture2D(mTexturePtr);
	}

	__host__ void SingleFrameColorTarget::CopyColorToDevice(int startIndex, int colorCount, ColorRGB * deviceColorPtr) const
	{
		RenderAPI* api = RenderAPI::GetRenderAPI();
		ColorRGB* mappedPtr = reinterpret_cast<ColorRGB*>(api->BeginReadTexture2D(mTexturePtr));
		MCopy(deviceColorPtr, mappedPtr + startIndex, colorCount * sizeof(ColorRGB), cudaMemcpyKind::cudaMemcpyDeviceToHost);
		api->EndReadTexture2D(mTexturePtr);
	}

	__host__ __device__ int SingleFrameColorTarget::GetFrameCount() const
	{
		return 1;
	}

	__host__ __device__ int SingleFrameColorTarget::GetFrameWidth() const
	{
		return mTextureResolution.x;
	}

	__host__ __device__ int SingleFrameColorTarget::GetFrameHeight() const
	{
		return mTextureResolution.y;
	}

	/*
		TODO:: MultiFrameColorTarget interface implementation
	*/
	__host__ void MultiFrameColorTarget::UpdateColorFromHost(int colorCount, const ColorRGB * colorPtr, int startIndex)
	{

	}

	__host__ void MultiFrameColorTarget::UpdateColorFromDevice(int colorCount, const ColorRGB * deviceColorPtr, int startIndex)
	{
	}

	__host__ void MultiFrameColorTarget::CopyColorToHost(int startIndex, int count, ColorRGB * colorPtr) const
	{
	}

	__host__ void MultiFrameColorTarget::CopyColorToDevice(int startIndex, int count, ColorRGB * deviceColorPtr) const
	{
	}

	__host__ __device__ int MultiFrameColorTarget::GetFrameCount() const
	{
		return 0;
	}

	__host__ __device__ int MultiFrameColorTarget::GetFrameWidth() const
	{
		return 0;
	}

	__host__ __device__ int MultiFrameColorTarget::GetFrameHeight() const
	{
		return 0;
	}

}
