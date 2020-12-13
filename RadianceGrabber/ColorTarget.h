#include "DataTypes.cuh"
#include "Util.h"

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

		__host__ virtual int GetFrameCount() const PURE;
		__host__ virtual int GetFrameWidth() const PURE;
		__host__ virtual int GetFrameHeight() const PURE;

		__host__ virtual void* GetHostColorBuffer() const PURE;
		__host__ virtual void* GetDeviceColorBuffer() const PURE;
		__host__ virtual void UploadDeviceToHost() PURE;
		__host__ virtual void UploadHostToDevice() PURE;
	};

	class SimpleColorTarget : public RadGrabber::IColorTarget
	{
	public:
		SimpleColorTarget(int w, int h) : w(w), h(h)
		{
			mBuffer = new ColorRGBA[w * h]; memset(mBuffer, 0, sizeof(ColorRGBA) * w * h); 
			mDeviceBuffer = (ColorRGBA*)MAllocDevice(sizeof(ColorRGBA) * w * h);
		}
		~SimpleColorTarget() { delete[] mBuffer; cudaFree(mDeviceBuffer);  }

		__host__ virtual void UpdateColorFromHost(int pixelIndex, int samplingCount, const ColorRGBA & color) override
		{
			//mBuffer[pixelIndex] = ColorRGBA((float)(pixelIndex % w) / w, (float)(pixelIndex / w) / h, 0.f, 0.f);
			mBuffer[pixelIndex] = (mBuffer[pixelIndex] * float(samplingCount - 1) / float(samplingCount)) + color / float(samplingCount);
		}
		__host__ virtual void CacheColorFromHost(int pixelIndex, int samplingCount, const ColorRGBA & color) override { }
		__host__ virtual void UploadColorFromHost() override { }

		__host__ virtual int GetFrameCount() const override { return 1; }
		__host__ virtual int GetFrameWidth() const override { return w; }
		__host__ virtual int GetFrameHeight() const override { return h; }

		__host__ virtual void* GetHostColorBuffer() const override { return mBuffer; }
		__host__ virtual void* GetDeviceColorBuffer() const override { return mDeviceBuffer; }
		__host__ virtual void UploadDeviceToHost() override
		{
			gpuErrchk(cudaMemcpy(mBuffer, mDeviceBuffer, sizeof(ColorRGBA) * w * h, cudaMemcpyKind::cudaMemcpyDeviceToHost));
		}
		__host__ virtual void UploadHostToDevice() override
		{
			gpuErrchk(cudaMemcpy(mDeviceBuffer, mBuffer, sizeof(ColorRGBA) * w * h, cudaMemcpyKind::cudaMemcpyHostToDevice));
		}

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
		ColorRGBA* mBuffer, *mDeviceBuffer;
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

		__host__ virtual int GetFrameCount() const override;
		__host__ virtual int GetFrameWidth() const override;
		__host__ virtual int GetFrameHeight() const override;

		__host__ virtual void* GetHostColorBuffer() const override;
		__host__ virtual void* GetDeviceColorBuffer() const override;
		__host__ virtual void UploadDeviceToHost() override;
		__host__ virtual void UploadHostToDevice() override;

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
		ColorRGBA* mDeviceColorBuffer;
	};

	/*
		TODO:: MultiFrameColorTarget class definition
	*/
	class MultiFrameColorTarget : public IColorTarget
	{
	public:
		__host__ MultiFrameColorTarget(int frameCount, Vector2i textureResolution);

		__host__ virtual void UpdateColorFromHost(int pixelIndex, int samplingCount, const ColorRGBA& color) override;
		__host__ virtual void CacheColorFromHost(int pixelIndex, int samplingCount, const ColorRGBA& color) override;
		__host__ virtual void UploadColorFromHost() override;

		// ColorTarget을(를) 통해 상속됨
		virtual __host__ int GetFrameCount() const override;
		virtual __host__ int GetFrameWidth() const override;
		virtual __host__ int GetFrameHeight() const override;

	};
}
