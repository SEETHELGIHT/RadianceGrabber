#include <cuda_runtime_api.h>
#include <cstdio>
#include <chrono>
#include <direct.h>
#include <csignal>
#include <thread>

#include "ConfigUtility.h"

#define RADIANCEGRABBER_REMOVE_LOG
#include "Util.h"
#include "ColorTarget.h"
#include "Aggregate.h"
#include "integrator.h"
#include "Marshal.cuh"
#include "Sample.cuh"
#include "DataTypes.cuh"
#include "DeviceConfig.h"
#include "Unity/RenderAPI.h"
#include "Pipeline.h"

namespace RadGrabber
{
	namespace Test
	{
		struct PathIntegratorSingleFrameSegment
		{
			SimpleColorTarget* target;
			FrameRequest* hreq;
			FrameInput* din;
			PathIntegrator* pi;
			LinearAggregate* lag;
			AcceleratedAggregate* aag;
			int threadCount;
			
			PathIntegratorSingleFrameSegment() : target(nullptr), hreq(nullptr), din(nullptr), pi(nullptr), lag(nullptr), aag(nullptr) {}

			void ClearSegments()
			{
				SAFE_HOST_DELETE(target);
				if (hreq) FreeHostMem(hreq);
				SAFE_HOST_DELETE(pi);

				if (din)	FreeDeviceMem(din);
				if (lag)	LinearAggregate::DestroyDeviceAggregate(lag);
				if (aag)	AcceleratedAggregate::DestroyDeviceAggregate(aag, threadCount);
			}

		} g_Seg;

		std::chrono::system_clock::time_point lastUpdateTime = std::chrono::system_clock::now();
		/*SimpleColorTarget* target;*/
		char fileDirAndName[513] = "";
		bool simpleScenePPMOpen = false;

		void update(int cnt)
		{
			printf("cnt::%d, %ld\n", cnt, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - lastUpdateTime).count());
			FILE* file = nullptr;
			fopen_s(&file, fileDirAndName, "wt");
			ASSERT(file);
			g_Seg.target->WritePPM(file);
			fclose(file);
			lastUpdateTime = std::chrono::system_clock::now();
		}

		void SignalHander(int sig)
		{
			g_Seg.ClearSegments();
		}

		int PathIntegratorAndLinearAggTest()
		{
			signal(SIGABRT, SignalHander);

			//SetFilePtr(stdout);
			SetBlockLog(1);

			char buf[129] = "SimpleScene";

			char buf2[257];
			sprintf(buf2, "../UnityExampleProjects/URPForRT/%s.framerequest", buf);
			FILE* fp = fopen(buf2, "rb");
			g_Seg.hreq = new FrameRequest();
			LoadFrameRequest(fp, &g_Seg.hreq, malloc);
			fclose(fp);

			if (Utility::ConfigUtility::IsImageConfigExist())
			{
				g_Seg.hreq->opt.maxSamplingCount = Utility::ConfigUtility::GetMaxSamplingCount();
				g_Seg.hreq->opt.maxDepth = Utility::ConfigUtility::GetPathMaxDepth();
				g_Seg.hreq->opt.resultImageResolution = Vector2i(Utility::ConfigUtility::GetImageWidth(), Utility::ConfigUtility::GetImageHeight());
			}

			g_Seg.hreq->opt.updateFunc = update;
			g_Seg.target = new SimpleColorTarget(g_Seg.hreq->opt.resultImageResolution.x, g_Seg.hreq->opt.resultImageResolution.y);
			g_Seg.pi = new PathIntegrator(g_Seg.hreq->opt);

			char dateStrPtr[257];
			time_t rawtime;
			struct tm timeinfo;
			time(&rawtime);
			localtime_s(&timeinfo, &rawtime);
 
			strftime(dateStrPtr, sizeof(dateStrPtr), "%Y-%m-%d_%H%M%S", &timeinfo);
			sprintf(fileDirAndName, "./%s_%s.ppm", buf, dateStrPtr);

			AllocateDeviceMem(g_Seg.hreq, &g_Seg.din);

			g_Seg.lag = LinearAggregate::GetAggregateDevice(&g_Seg.hreq->input, g_Seg.din, 0);

			OptimalLaunchParam param;
			GetOptimalBlockAndThreadDim(0, param);
			g_Seg.pi->Render(*g_Seg.lag, g_Seg.hreq->input, *g_Seg.din, g_Seg.hreq->opt, *g_Seg.target, param);

			g_Seg.ClearSegments();

			gpuErrchk(cudaDeviceReset());

			printf("Open ppm file::%s", fileDirAndName);

			char cCurrentPath[2024];
			_getcwd(cCurrentPath, sizeof(cCurrentPath));
			strcat_s(cCurrentPath, sizeof(cCurrentPath), fileDirAndName + 1);
			strcat_s(cCurrentPath, sizeof(cCurrentPath), " & ");
			system(cCurrentPath);

			return 0;
		}

		int PathIntegratorAndAccelAggTest()
		{
			signal(SIGABRT, SignalHander);

			//SetFilePtr(stdout);
			SetBlockLog(1);

			char buf[129] = "SimpleScene";

			char buf2[257];
			sprintf(buf2, "../UnityExampleProjects/URPForRT/%s.framerequest", buf);
			FILE* fp = fopen(buf2, "rb");
			g_Seg.hreq = new FrameRequest();
			LoadFrameRequest(fp, &g_Seg.hreq, malloc);
			fclose(fp);

			if (Utility::ConfigUtility::IsImageConfigExist())
			{
				g_Seg.hreq->opt.maxSamplingCount = Utility::ConfigUtility::GetMaxSamplingCount();
				g_Seg.hreq->opt.maxDepth = Utility::ConfigUtility::GetPathMaxDepth();
				g_Seg.hreq->opt.resultImageResolution = Vector2i(Utility::ConfigUtility::GetImageWidth(), Utility::ConfigUtility::GetImageHeight());
			}

			g_Seg.hreq->opt.updateFunc = update;

			g_Seg.target = new SimpleColorTarget(g_Seg.hreq->opt.resultImageResolution.x, g_Seg.hreq->opt.resultImageResolution.y);
			g_Seg.pi = new PathIntegrator(g_Seg.hreq->opt);

			{
				char dateStrPtr[257];
				time_t rawtime;
				struct tm timeinfo;
				time(&rawtime);
				localtime_s(&timeinfo, &rawtime);

				strftime(dateStrPtr, sizeof(dateStrPtr), "%Y-%m-%d_%H%M%S", &timeinfo);
				sprintf(fileDirAndName, "./%s_%s.ppm", buf, dateStrPtr);
			}

			AllocateDeviceMem(g_Seg.hreq, &g_Seg.din);

			g_Seg.threadCount = 2048;
			g_Seg.aag = AcceleratedAggregate::GetAggregateDevice(&g_Seg.hreq->input, g_Seg.din, 0, g_Seg.threadCount);

			OptimalLaunchParam param;
			GetOptimalBlockAndThreadDim(0, param);
			g_Seg.pi->Render(*g_Seg.aag, g_Seg.hreq->input, *g_Seg.din, g_Seg.hreq->opt, *g_Seg.target, param);

			g_Seg.ClearSegments();

			gpuErrchk(cudaDeviceReset());

			printf("Open ppm file::%s", fileDirAndName);

			char cCurrentPath[2024];
			_getcwd(cCurrentPath, sizeof(cCurrentPath));
			strcat_s(cCurrentPath, sizeof(cCurrentPath), fileDirAndName + 1);
			strcat_s(cCurrentPath, sizeof(cCurrentPath), " & ");
			system(cCurrentPath);

			return 0;
		}
	}
}