#include "ColorTarget.h"
#include "Marshal.cuh"
#include "DLLEntry.h"
#include "Pipeline.h"
#include "Image.h"
#include "ConfigUtility.h"
#include "DeviceConfig.h"

namespace RadGrabber
{
	namespace Test
	{
		int SIngleFrameGenThreadManagementTest()
		{
			SetFilePtr(stdout);

			char buf[129] = "SimpleScene";
			char buf2[257];
			sprintf(buf2, "../UnityExampleProjects/URPForRT/%s.framerequest", buf);
			FILE* fp = fopen(buf2, "rb");
			FrameRequest* req = new FrameRequest();
			LoadFrameRequest(fp, &req, malloc);
			fclose(fp);

			FrameRequestMarshal* mar = new FrameRequestMarshal();
			mar->input = req->input.in;
			mar->opt = req->opt;
			mar->output = req->output;

			SimpleColorTarget* target = new SimpleColorTarget(mar->opt.resultImageResolution.x, mar->opt.resultImageResolution.y);
			mar->output.pixelBuffer = target->GetHostColorBuffer();
			GenerateSingleFrameIncremental(mar);

			Sleep(10000);

			StopGenerateSingleFrame();

			while (IsSingleFrameGenerating());

			delete mar;
			delete target;

			return 0;
		}

		MultiFrameRequest* mreq;
		void MultiFrmaeUpdateFrameFunc(int frameIndex, void* rgbaPtr)
		{
			char buffer[4096];
			sprintf(buffer, "./temp/%d", frameIndex);
			ImageWrite(buffer, ".png", (ColorRGBA*)rgbaPtr, mreq->opt.resultImageResolution.x, mreq->opt.resultImageResolution.y);

			if (frameIndex + 1 == mreq->input.GetCount())
				VideoWrite("./temp/%s", ".png", "./temp/SimpleScene.avi", mreq->opt.resultImageResolution.x, mreq->opt.resultImageResolution.y, 60.0, mreq->input.GetCount());
		}
		int MultiFrameGenThreadManagementTest()
		{
			SetFilePtr(stdout);

			char buf[129] = "SimpleScene";
			char buf2[257];
			sprintf(buf2, "../UnityExampleProjects/URPForRT/%s.multiframerequest", buf);
			FILE* fp = fopen(buf2, "rb");
			mreq = new MultiFrameRequest();
			LoadMultiFrameRequest(fp, &mreq, malloc);
			fclose(fp);

			mreq->opt.updateFrameFunc = MultiFrmaeUpdateFrameFunc;

			OptimalLaunchParam param;
			GetOptimalBlockAndThreadDim(0, param);

			Utility::FramePathTestConfig pathConfig;
			pathConfig.RefreshValues();

			if (!pathConfig.imageConfigIgnore)
			{
				pathConfig.Set(mreq->opt, param.threadCountinBlock, param.blockCountInGrid);
				mreq->input.startIndex = pathConfig.startIndex;
				mreq->input.endCount = pathConfig.endCount;
			}

			SimpleColorTarget* target = new SimpleColorTarget(mreq->opt.resultImageResolution.x, mreq->opt.resultImageResolution.y);
			mreq->output.pixelBuffer = target->GetHostColorBuffer();
			GenerateMultiFrameIncrementalRunitme(mreq);

			Sleep(30000);

			StopGenerateSingleFrame();

			while (IsMultiFrameGenerating());

			delete target;

			puts("MultiFrameGenThreadManagementTest() END");

			return 0;
		}
	}
}
