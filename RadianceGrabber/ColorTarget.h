#include "DataTypes.cuh"

#pragma once

namespace RadGrabber
{
	struct ColorRGB;

	class IColorTarget abstract
	{
	public:
		virtual ~IColorTarget() {}

		__host__ virtual void UpdateColorFromHost(int pixelIndex, int samplingCount, const ColorRGBA& color) PURE;
		__host__ virtual void CacheColorFromHost(int pixelIndex, int samplingCount, const ColorRGBA& color) PURE;
		__host__ virtual void UploadColorFromHost() PURE;

		__host__ __device__ virtual int GetFrameCount() const PURE;
		__host__ __device__ virtual int GetFrameWidth() const PURE;
		__host__ __device__ virtual int GetFrameHeight() const PURE;
	};

	class SimpleColorTarget : public RadGrabber::IColorTarget
	{
	public:
		SimpleColorTarget(int w, int h) : w(w), h(h) { mBuffer = new ColorRGBA[w * h]; memset(mBuffer, 0, sizeof(ColorRGBA) * w * h); }
		~SimpleColorTarget() { delete[] mBuffer; }

		virtual __host__ void UpdateColorFromHost(int pixelIndex, int samplingCount, const ColorRGBA & color) override
		{
			//mBuffer[pixelIndex] = ColorRGBA((float)(pixelIndex % w) / w, (float)(pixelIndex / w) / h, 0.f, 0.f);
			mBuffer[pixelIndex] = (mBuffer[pixelIndex] * (samplingCount - 1) / samplingCount) + color / samplingCount;
		}
		virtual __host__ void CacheColorFromHost(int pixelIndex, int samplingCount, const ColorRGBA & color) override { }
		virtual __host__ void UploadColorFromHost() override { }

		__host__ __device__ virtual int GetFrameCount() const override { return 1; }
		__host__ __device__ virtual int GetFrameWidth() const override { return w; }
		__host__ __device__ virtual int GetFrameHeight() const override { return h; }

		void WritePPM(FILE *fp)
		{
			fprintf(fp, "P3\n%d %d 255\n", GetFrameWidth(), GetFrameHeight());
			for (int i = 0; i < GetFrameHeight(); i++, fprintf(fp, "\n"))
				for (int j = 0; j < GetFrameWidth(); j++)
					fprintf(
						fp, 
						"%d %d %d ", 
						min((int)(mBuffer[(h - i - 1) * GetFrameWidth() + j].r * 255), 255), 
						min((int)(mBuffer[(h - i - 1) * GetFrameWidth() + j].g * 255), 255),
						min((int)(mBuffer[(h - i - 1) * GetFrameWidth() + j].b * 255), 255)
					);
		}

	public:
		ColorRGBA* mBuffer;
		int w, h;
	};

	class SingleFrameColorTarget : public IColorTarget
	{
	public:
		__host__ SingleFrameColorTarget(int colorBufferCount, Vector2i textureResolution, void* texturePtr);
		__host__ ~SingleFrameColorTarget();

		__host__ virtual void UpdateColorFromHost(int pixelIndex, int samplingCount, const ColorRGBA& color) override;
		__host__ virtual void CacheColorFromHost(int pixelIndex, int samplingCount, const ColorRGBA& color) override;
		__host__ virtual void UploadColorFromHost() override;

		__host__ __device__ virtual int GetFrameCount() const override;
		__host__ __device__ virtual int GetFrameWidth() const override;
		__host__ __device__ virtual int GetFrameHeight() const override;

	private:
		Vector2i mTextureResolution;
		void* mTexturePtr;

		struct ColorBuffer
		{
			int pixelIndex;
			int samplingCount;
			ColorRGBA color;
		};

		int mColorBufferCount;
		int mStartColorIndex;
		int mNextColorIndex;

		ColorBuffer* mHostColorBuffer;
	};

	/*
		TODO:: MultiFrameColorTarget class definition
	*/
	class MultiFrameColorTarget : public IColorTarget
	{
	public:
		__host__ MultiFrameColorTarget(int frameCount, Vector2i textureResolution);

		// ColorTarget을(를) 통해 상속됨
		virtual __host__ __device__ int GetFrameCount() const override;
		virtual __host__ __device__ int GetFrameWidth() const override;
		virtual __host__ __device__ int GetFrameHeight() const override;

	};
}
