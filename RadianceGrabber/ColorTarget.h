#include "DataTypes.cuh"

#pragma once

namespace RadGrabber
{
	struct ColorRGB;

	class IColorTarget abstract
	{
	public:
		__host__ virtual void UpdateColorFromHost(int colorCount, const ColorRGB* colorPtr, int startIndex) PURE;
		__host__ virtual void UpdateColorFromDevice(int colorCount, const ColorRGB* deviceColorPtr, int startIndex) PURE;
		__host__ virtual void CopyColorToHost(int startIndex, int count, ColorRGB* colorPtr) const PURE;
		__host__ virtual void CopyColorToDevice(int startIndex, int count, ColorRGB* deviceColorPtr) const PURE;

		__host__ __device__ virtual int GetFrameCount() const PURE;
		__host__ __device__ virtual int GetFrameWidth() const PURE;
		__host__ __device__ virtual int GetFrameHeight() const PURE;
	};

	class SingleFrameColorTarget : public IColorTarget
	{
	public:
		__host__ SingleFrameColorTarget(Vector2i textureResolution, void* texturePtr);
		__host__ ~SingleFrameColorTarget();

		// ColorTarget을(를) 통해 상속됨
		virtual __host__ void UpdateColorFromHost(int colorCount, const ColorRGB * colorPtr, int startIndex) override;
		virtual __host__ void UpdateColorFromDevice(int colorCount, const ColorRGB * deviceColorPtr, int startIndex) override;
		virtual __host__ void CopyColorToHost(int startIndex, int count, ColorRGB * colorPtr) const override;
		virtual __host__ void CopyColorToDevice(int startIndex, int count, ColorRGB * deviceColorPtr) const override;

		virtual __host__ __device__ int GetFrameCount() const override;
		virtual __host__ __device__ int GetFrameWidth() const override;
		virtual __host__ __device__ int GetFrameHeight() const override;

	private:
		Vector2i mTextureResolution;
		void* mTexturePtr;
	};

	/*
		TODO:: MultiFrameColorTarget class definition
	*/
	class MultiFrameColorTarget : public IColorTarget
	{
	public:
		__host__ MultiFrameColorTarget(int frameCount, Vector2i textureResolution);

		// ColorTarget을(를) 통해 상속됨
		virtual __host__ void UpdateColorFromHost(int colorCount, const ColorRGB * colorPtr, int startIndex) override;
		virtual __host__ void UpdateColorFromDevice(int colorCount, const ColorRGB * deviceColorPtr, int startIndex) override;
		virtual __host__ void CopyColorToHost(int startIndex, int count, ColorRGB * colorPtr) const override;
		virtual __host__ void CopyColorToDevice(int startIndex, int count, ColorRGB * deviceColorPtr) const override;

		virtual __host__ __device__ int GetFrameCount() const override;
		virtual __host__ __device__ int GetFrameWidth() const override;
		virtual __host__ __device__ int GetFrameHeight() const override;
	};
}
