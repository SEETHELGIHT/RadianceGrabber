#include "ColorTarget.h"
#include "Util.h"
#include "Unity/RenderAPI.h"

namespace RadGrabber
{
	__host__ SingleFrameColorTarget::SingleFrameColorTarget(int colorBufferCount, Vector2i textureResolution, void * texturePtr) : 
		mTextureResolution(textureResolution), mTexturePtr(texturePtr), mStartColorIndex(0), mNextColorIndex(0)
	{
		mColorBufferCount = colorBufferCount;
		mHostColorBuffer = MAllocHost<ColorBuffer>(mColorBufferCount);
		MSet(mHostColorBuffer, 0, sizeof(SingleFrameColorTarget::ColorBuffer) * mColorBufferCount);
		mDeviceColorBuffer = (ColorRGBA*)MAllocDevice(sizeof(ColorRGBA) * textureResolution.x * textureResolution.y);
	}

	__host__ SingleFrameColorTarget::~SingleFrameColorTarget()
	{ 
		SAFE_HOST_DELETE(mHostColorBuffer);
	}

	__host__ int SingleFrameColorTarget::GetFrameCount() const
	{
		return 1;
	}

	__host__ int SingleFrameColorTarget::GetFrameWidth() const
	{
		return mTextureResolution.x;
	}

	__host__ int SingleFrameColorTarget::GetFrameHeight() const
	{
		return mTextureResolution.y;
	}

	__host__ void SingleFrameColorTarget::UpdateColorFromHost(int pixelIndex, int samplingCount, const ColorRGBA& color)
	{
		ColorRGBA* mappedPtr = reinterpret_cast<ColorRGBA*>(mTexturePtr);
		mappedPtr[pixelIndex] = (mappedPtr[pixelIndex] * (samplingCount - 1) / samplingCount) + color / samplingCount;
		mappedPtr[pixelIndex].a = 1;
	}
	__host__ void SingleFrameColorTarget::CacheColorFromHost(int pixelIndex, int samplingCount, const ColorRGBA& color)
	{
		int colorIndex = ++mNextColorIndex;
		mHostColorBuffer[colorIndex].pixelIndex = pixelIndex;
		mHostColorBuffer[colorIndex].samplingCount = samplingCount;
		mHostColorBuffer[colorIndex].color = color;
	}
	__host__ void SingleFrameColorTarget::UploadColorFromHost()
	{
		ColorRGBA* mappedPtr = reinterpret_cast<ColorRGBA*>(mTexturePtr);
		for (int i = mStartColorIndex; i != mNextColorIndex; i = (i + 1) % mColorBufferCount)
			mappedPtr[mHostColorBuffer[i].pixelIndex] = 
				mappedPtr[mHostColorBuffer[i].pixelIndex] * (float(mHostColorBuffer[i].samplingCount-1) / (mHostColorBuffer[i].samplingCount)) + 
				mHostColorBuffer[i].color / mHostColorBuffer[i].samplingCount;

		mStartColorIndex = mNextColorIndex = 0;
	}

	__host__ void* SingleFrameColorTarget::GetHostColorBuffer() const
	{
		return mTexturePtr;
	}
	__host__ void* SingleFrameColorTarget::GetDeviceColorBuffer() const
	{
		return mDeviceColorBuffer;
	}
	__host__ void SingleFrameColorTarget::UploadDeviceToHost()
	{
		gpuErrchk(cudaMemcpy(mTexturePtr, mDeviceColorBuffer, sizeof(ColorRGBA) * mTextureResolution.x * mTextureResolution.y, cudaMemcpyKind::cudaMemcpyDeviceToHost));
	}
	__host__ void SingleFrameColorTarget::UploadHostToDevice()
	{
		gpuErrchk(cudaMemcpy(mDeviceColorBuffer, mTexturePtr, sizeof(ColorRGBA) * mTextureResolution.x * mTextureResolution.y, cudaMemcpyKind::cudaMemcpyHostToDevice));
	}

	/*
		TODO:: MultiFrameColorTarget interface implementation
	*/

	__host__ int MultiFrameColorTarget::GetFrameCount() const
	{
		return 0;
	}

	__host__ int MultiFrameColorTarget::GetFrameWidth() const
	{
		return 0;
	}

	__host__ int MultiFrameColorTarget::GetFrameHeight() const
	{
		return 0;
	}

}
