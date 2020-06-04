#include "ColorTarget.h"
#include "Marshal.cuh"
#include "DLLEntry.h"
#include "Pipeline.h"

namespace RadGrabber
{
	namespace Test
	{
		class DummyColorTarget2 : public RadGrabber::IColorTarget
		{
		public:
			DummyColorTarget2(int x, int y) : x(x), y(y)
			{
				mBuffer = new ColorRGBA[x * y];
			}
			~DummyColorTarget2()
			{
				delete[] mBuffer;
			}

			virtual __host__ void UpdateColorFromHost(int pixelIndex, int samplingCount, const ColorRGBA & color) override
			{
				mBuffer[pixelIndex] = (mBuffer[pixelIndex] * (samplingCount - 1) / samplingCount) + color / samplingCount;
			}
			virtual __host__ void CacheColorFromHost(int pixelIndex, int samplingCount, const ColorRGBA & color) override { }
			virtual __host__ void UploadColorFromHost() override { }

			__host__ __device__ virtual int GetFrameCount() const override { return 1; }
			__host__ __device__ virtual int GetFrameWidth() const override { return x; }
			__host__ __device__ virtual int GetFrameHeight() const override { return y; }

			void WritePPM(FILE *fp)
			{
				fprintf(fp, "P3\n%d %d 255\n", GetFrameWidth(), GetFrameHeight());
				for (int i = 0; i < GetFrameHeight(); i++, fprintf(fp, "\n"))
					for (int j = 0; j < GetFrameWidth(); j++)
						fprintf(fp, "%d %d %d ", (int)(mBuffer[i * GetFrameWidth() + j].r * 255), (int)(mBuffer[i * GetFrameWidth() + j].g * 255), (int)(mBuffer[i * GetFrameWidth() + j].b * 255));
			}

		public:
			int x, y;
			ColorRGBA* mBuffer;
		};

		int ThreadManagementTest()
		{
			FILE* fp;
			fopen_s(&fp, "./SimpleScene.framerequest", "rb");
			FrameRequest* req = new FrameRequest();
			LoadFrameRequest(fp, &req, malloc);
			fclose(fp);

			FrameRequestMarshal* mar = new FrameRequestMarshal();
			mar->input = req->input.in;
			mar->opt = req->opt;
			mar->output = req->output;

			DummyColorTarget2* target = new DummyColorTarget2(mar->opt.resultImageResolution.x, mar->opt.resultImageResolution.y);

			GenerateSingleFrameIncrementalTest(mar, target);

			Sleep(10000);

			StopGenerateSingleFrame();

			while (IsSingleFrameGenerating());

			return 0;
		}
	}
}
