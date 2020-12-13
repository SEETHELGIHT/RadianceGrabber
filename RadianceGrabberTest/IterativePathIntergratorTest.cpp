#include <cuda_runtime_api.h>
#define _CRT_SECURE_NO_WARNINGS
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

#include "AcceleratedAggregate.h"
#include "IterativePathIntegrator.h"

namespace RadGrabber
{
	namespace Test
	{
		struct IterativePathIntegratorSingleFrameSegment
		{
			SimpleColorTarget* target;
			FrameRequest* hreq;
			FrameInput* din;
			IterativePathIntegrator* pi;
			AcceleratedAggregate* aag;
			int threadCount;

			IterativePathIntegratorSingleFrameSegment() : target(nullptr), hreq(nullptr), din(nullptr), pi(nullptr), aag(nullptr) {}

			void ClearSegments()
			{
				SAFE_HOST_DELETE(target);

				if (hreq)
				{
					FreeHostFrameRequest(hreq);
					hreq = nullptr;
				}

				SAFE_HOST_DELETE(pi);

				if (din)
				{
					FreeDeviceFrameRequest(din);
					din = nullptr;
				}
				if (aag)
				{
					AcceleratedAggregate::DestroyDeviceAggregate(aag);
					aag = nullptr;
				}
			}

		};

		struct IterativePathIntegratorSingleFrameSegment g_iiSeg;
		extern std::chrono::system_clock::time_point lastUpdateTime;
		extern char fileDirAndName[513];
		extern bool simpleScenePPMOpen;

		const IIteratableAggregate* Get(int m) { return g_iiSeg.aag; }

		void update(int cnt, void* rgbaptr);

		void ISignalHander(int sig)
		{
			g_iiSeg.ClearSegments();
		}

		int IterativePathIntegratorAndAccelAggTest()
		{
			signal(SIGABRT, ISignalHander);

			SetFilePtr(stdout);
			SetBlockLog(1);

			TimeProfiler init(GetFilePtr(), "PT Test::Param Init");

			char buf[129] = "SimpleScene";

			char buf2[257];
			sprintf(buf2, "../UnityExampleProjects/URPForRT/%s.framerequest", buf);
			FILE* fp = fopen(buf2, "rb");
			g_iiSeg.hreq = new FrameRequest();
			LoadFrameRequest(fp, &g_iiSeg.hreq, malloc);
			fclose(fp);

			OptimalLaunchParam param;
			GetOptimalBlockAndThreadDim(0, param);

			Utility::FramePathTestConfig pathConfig;
			pathConfig.RefreshValues();

			if (!pathConfig.imageConfigIgnore)
			{
				pathConfig.Set(g_iiSeg.hreq->opt, param.threadCountinBlock, param.blockCountInGrid);
			}

			g_iiSeg.hreq->opt.updateFunc = nullptr;
			g_iiSeg.hreq->opt.updateFrameFunc = update;

			init.Print();

			TimeProfiler objectInit(GetFilePtr(), "IPT Test::Object Init");

			g_iiSeg.target = new SimpleColorTarget(g_iiSeg.hreq->opt.resultImageResolution.x, g_iiSeg.hreq->opt.resultImageResolution.y);
			g_iiSeg.pi = new IterativePathIntegrator(g_iiSeg.hreq->opt, param);

			{
				char dateStrPtr[257];
				time_t rawtime;
				struct tm timeinfo;
				time(&rawtime);
				localtime_s(&timeinfo, &rawtime);

				strftime(dateStrPtr, sizeof(dateStrPtr), "%Y-%m-%d_%H%M%S", &timeinfo);
				sprintf(fileDirAndName, "./Images/%s_%s.ppm", buf, dateStrPtr);
			}

			AllocateDeviceFrameRequest(g_iiSeg.hreq, &g_iiSeg.din);

			g_iiSeg.threadCount = param.threadCountinBlock.x * param.blockCountInGrid.x;
			g_iiSeg.aag = AcceleratedAggregate::GetAggregateDevice(&g_iiSeg.hreq->input, g_iiSeg.din, 0, g_iiSeg.threadCount);

			objectInit.Print();

			{
				TimeProfiler render(GetFilePtr(), "IPT Test::RenderIncremental");
				g_iiSeg.pi->RenderIncremental(Get, HostDevicePair<IMultipleInput*>(&g_iiSeg.hreq->input, g_iiSeg.din), *g_iiSeg.target, g_iiSeg.hreq->opt, param);
				render.Print();
			}

			{
				TimeProfiler t(GetFilePtr(), "IPT Test::ClearSegments");
				g_iiSeg.ClearSegments();
				t.Print();
			}


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