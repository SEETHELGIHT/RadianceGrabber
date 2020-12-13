#include <cuda_runtime_api.h>
#define _CRT_SECURE_NO_WARNINGS
#include <cstdio>
#include <chrono>
#include <direct.h>
#include <csignal>
#include <thread>
#include <string>

#include "ConfigUtility.h"

#define RADIANCEGRABBER_REMOVE_LOG
#include "Util.h"
#include "ColorTarget.h"
#include "Marshal.cuh"
#include "Sample.cuh"
#include "DataTypes.cuh"
#include "DeviceConfig.h"
#include "Unity/RenderAPI.h"
#include "Pipeline.h"

#include "LinearAggregate.h"
#include "AcceleratedAggregate.h"
#include "PathIntegrator.h"

#include "Image.h"

namespace RadGrabber
{
	namespace Test
	{
		struct PathIntegratorSingleFrameSegment
		{
			SimpleColorTarget* target;
			FrameRequest* hreq;
			MultiFrameRequest* hmreq;
			FrameInput* din;
			MultiFrameInput* dmin;
			PathIntegrator* pi;
			LinearAggregate* lag;
			AcceleratedAggregate* aag;
			AcceleratedAggregate* aagArray;
			int threadCount;
			
			PathIntegratorSingleFrameSegment() : target(nullptr), hreq(nullptr), din(nullptr), pi(nullptr), lag(nullptr), aag(nullptr) {}

			void ClearSegments()
			{
				SAFE_HOST_DELETE(target);

				if (hreq)
				{
					FreeHostFrameRequest(hreq);
					hreq = nullptr;
				}
				if (hmreq)
				{
					FreeHostMultiFrameRequest(hmreq);
					hreq = nullptr;
				}
					
				SAFE_HOST_DELETE(pi);

				if (din)
				{
					FreeDeviceFrameRequest(din);
					din = nullptr;
				}
				if (dmin)
				{
					FreeDeviceMultiFrameRequest(dmin);
					dmin = nullptr;
				}
				if (lag)
				{
					LinearAggregate::DestroyDeviceAggregate(lag);
					lag = nullptr;
				}
				if (aag)
				{
					AcceleratedAggregate::DestroyDeviceAggregate(aag);
					aag = nullptr;
				}
				if (aagArray)
				{
					AcceleratedAggregate::DestroyDeviceAggregate(hmreq->input.in.mutableInputLen, aagArray);
					aagArray = nullptr;
				}
			}

		} g_Seg;

		const IAggregate* GetAggregate(int mutableIndex)
		{
			if (g_Seg.lag)
				return g_Seg.lag;
			else if (g_Seg.aag)
				return g_Seg.aag;
			else
				return g_Seg.aagArray + mutableIndex;
		}

		std::chrono::system_clock::time_point lastUpdateTime = std::chrono::system_clock::now();
		/*SimpleColorTarget* target;*/
		char fileDirAndName[513] = "";
		bool simpleScenePPMOpen = false;

		void update(int cnt, void* rgbaptr)
		{
			printf("frame::%d, %lld\n", cnt, std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now() - lastUpdateTime).count());
			FILE* file = nullptr;
			
			for(int tryCount = 1; !file; tryCount++, std::this_thread::sleep_for(std::chrono::duration<int>(1)))
			{
				printf("Try to open..(%d)\n", tryCount);
				fopen_s(&file, fileDirAndName, "wt");
			}

			g_Seg.target->UploadDeviceToHost();
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

			SetFilePtr(stdout);
			//SetBlockLog(1);

			Utility::FramePathTestConfig pathConfig;
			pathConfig.RefreshValues();
			Utility::TestProjectConfig projConfig;
			projConfig.RefreshValues();

			FILE* fp = fopen(projConfig.frameRequestPath, "rb");
			g_Seg.hreq = new FrameRequest();
			LoadFrameRequest(fp, &g_Seg.hreq, malloc);
			fclose(fp);

			OptimalLaunchParam param;
			GetOptimalBlockAndThreadDim(0, param);

			if (!pathConfig.imageConfigIgnore)
				pathConfig.Set(g_Seg.hmreq->opt, param.threadCountinBlock, param.blockCountInGrid);

			g_Seg.hreq->opt.updateFunc = nullptr;
			g_Seg.hreq->opt.updateFrameFunc = update;
			g_Seg.target = new SimpleColorTarget(g_Seg.hreq->opt.resultImageResolution.x, g_Seg.hreq->opt.resultImageResolution.y);
			g_Seg.pi = new PathIntegrator(param, g_Seg.hreq->opt.resultImageResolution, (PathIntegratorPixelContent)pathConfig.pixelContent, true);

			{
				char dateStrPtr[257];
				time_t rawtime;
				struct tm timeinfo;
				time(&rawtime);
				localtime_s(&timeinfo, &rawtime);

				strftime(dateStrPtr, sizeof(dateStrPtr), "%Y-%m-%d_%H%M%S", &timeinfo);

				std::string requestPath = std::string(projConfig.frameRequestPath);

				int commaIndex = requestPath.find_last_of('.');
				std::string fileName;

				if (commaIndex == std::string::npos)
					fileName = requestPath.substr(requestPath.find_last_of('/') + 1);
				else
					fileName = requestPath.substr(requestPath.find_last_of('/') + 1, commaIndex - 1);

				sprintf(fileDirAndName, "./Images/%s_%s.ppm", fileName.c_str(), dateStrPtr);
			}

			AllocateDeviceFrameRequest(g_Seg.hreq, &g_Seg.din);

			g_Seg.lag = LinearAggregate::GetAggregateDevice(&g_Seg.hreq->input, g_Seg.din, 0);

			if (pathConfig.incrementalMode)
				g_Seg.pi->RenderIncremental(GetAggregate, HostDevicePair<IMultipleInput*>(&g_Seg.hreq->input, g_Seg.din), *g_Seg.target, g_Seg.hreq->opt, param);
			else
				g_Seg.pi->RenderStraight(GetAggregate, HostDevicePair<IMultipleInput*>(&g_Seg.hreq->input, g_Seg.din), *g_Seg.target, g_Seg.hreq->opt, param);

			gpuErrchk(cudaDeviceSynchronize());

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

			SetFilePtr(stdout);
			//SetBlockLog(1);

			TimeProfiler init(GetFilePtr(), "PT Test::Param Init");

			Utility::FramePathTestConfig pathConfig;
			pathConfig.RefreshValues();
			Utility::TestProjectConfig projConfig;
			projConfig.RefreshValues();

			FILE* fp = fopen(projConfig.frameRequestPath, "rb");
			g_Seg.hreq = new FrameRequest();
			LoadFrameRequest(fp, &g_Seg.hreq, malloc);
			fclose(fp);

			OptimalLaunchParam param;
			GetOptimalBlockAndThreadDim(0, param);

			if (!pathConfig.imageConfigIgnore)
			{
				pathConfig.Set(g_Seg.hreq->opt, param.threadCountinBlock, param.blockCountInGrid);
			}

			g_Seg.hreq->opt.updateFunc = nullptr;
			g_Seg.hreq->opt.updateFrameFunc = update;

			init.Print();

			TimeProfiler objectInit(GetFilePtr(), "PT Test::Object Init");

			g_Seg.target = new SimpleColorTarget(g_Seg.hreq->opt.resultImageResolution.x, g_Seg.hreq->opt.resultImageResolution.y);
			g_Seg.pi = new PathIntegrator(param, g_Seg.hreq->opt.resultImageResolution, (PathIntegratorPixelContent)pathConfig.pixelContent, true);

			{
				char dateStrPtr[257];
				time_t rawtime;
				struct tm timeinfo;
				time(&rawtime);
				localtime_s(&timeinfo, &rawtime);

				strftime(dateStrPtr, sizeof(dateStrPtr), "%Y-%m-%d_%H%M%S", &timeinfo);

				std::string requestPath = std::string(projConfig.frameRequestPath);

				int commaIndex = requestPath.find_last_of('.');
				std::string fileName;

				if (commaIndex == std::string::npos)
					fileName = requestPath.substr(requestPath.find_last_of('/') + 1);
				else
					fileName = requestPath.substr(requestPath.find_last_of('/') + 1, commaIndex - 1);

				sprintf(fileDirAndName, "./Images/%s_%s.ppm", fileName.c_str(), dateStrPtr);
			}

			AllocateDeviceFrameRequest(g_Seg.hreq, &g_Seg.din);

			g_Seg.threadCount = param.GetMaxThreadCount();
			g_Seg.aag = AcceleratedAggregate::GetAggregateDevice(&g_Seg.hreq->input, g_Seg.din, 0, g_Seg.threadCount);

			objectInit.Print();

			{
				TimeProfiler render(GetFilePtr(), "PT Test::RenderIncremental");
				if (pathConfig.incrementalMode)
					g_Seg.pi->RenderIncremental(GetAggregate, HostDevicePair<IMultipleInput*>(&g_Seg.hreq->input, g_Seg.din), *g_Seg.target, g_Seg.hreq->opt, param);
				else
					g_Seg.pi->RenderStraight(GetAggregate, HostDevicePair<IMultipleInput*>(&g_Seg.hreq->input, g_Seg.din), *g_Seg.target, g_Seg.hreq->opt, param);
				render.Print();
			}

			{
				TimeProfiler t(GetFilePtr(), "PT Test::ClearSegments");
				g_Seg.ClearSegments();
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

		int PathIntegratorStraigntAndIncremental()
		{
			signal(SIGABRT, SignalHander);

			SetFilePtr(stdout);
			//SetBlockLog(1);

			Utility::FramePathTestConfig pathConfig;
			pathConfig.RefreshValues();
			Utility::TestProjectConfig projConfig;
			projConfig.RefreshValues();

			FILE* fp = fopen(projConfig.frameRequestPath, "rb");
			g_Seg.hreq = new FrameRequest();
			LoadFrameRequest(fp, &g_Seg.hreq, malloc);
			fclose(fp);

			OptimalLaunchParam param;
			GetOptimalBlockAndThreadDim(0, param);

			if (!pathConfig.imageConfigIgnore)
				pathConfig.Set(g_Seg.hreq->opt, param.threadCountinBlock, param.blockCountInGrid);

			g_Seg.hreq->opt.updateFunc = nullptr;
			g_Seg.hreq->opt.updateFrameFunc = update;

			g_Seg.target = new SimpleColorTarget(g_Seg.hreq->opt.resultImageResolution.x, g_Seg.hreq->opt.resultImageResolution.y);
			g_Seg.pi = new PathIntegrator(param, g_Seg.hreq->opt.resultImageResolution, (PathIntegratorPixelContent)pathConfig.pixelContent, true);

			{
				char dateStrPtr[257];
				time_t rawtime;
				struct tm timeinfo;
				time(&rawtime);
				localtime_s(&timeinfo, &rawtime);

				strftime(dateStrPtr, sizeof(dateStrPtr), "%Y-%m-%d_%H%M%S", &timeinfo);

				std::string requestPath = std::string(projConfig.frameRequestPath);

				int commaIndex = requestPath.find_last_of('.');
				std::string fileName;

				if (commaIndex == std::string::npos)
					fileName = requestPath.substr(requestPath.find_last_of('/') + 1);
				else
					fileName = requestPath.substr(requestPath.find_last_of('/') + 1, commaIndex - 1);

				sprintf(fileDirAndName, "./Images/%s_%s.ppm", fileName.c_str(), dateStrPtr);
			}

			AllocateDeviceFrameRequest(g_Seg.hreq, &g_Seg.din);

			g_Seg.threadCount = param.GetMaxThreadCount();
			g_Seg.aag = AcceleratedAggregate::GetAggregateDevice(&g_Seg.hreq->input, g_Seg.din, 0, g_Seg.threadCount);

			{
				TimeProfiler p = TimeProfiler("RenderIncremental");
				g_Seg.pi->RenderIncremental(GetAggregate, HostDevicePair<IMultipleInput*>(&g_Seg.hreq->input, g_Seg.din), *g_Seg.target, g_Seg.hreq->opt, param);
			}

			{
				TimeProfiler p = TimeProfiler("RenderStraight");
				g_Seg.pi->RenderStraight(GetAggregate, HostDevicePair<IMultipleInput*>(&g_Seg.hreq->input, g_Seg.din), *g_Seg.target, g_Seg.hreq->opt, param);
			}

			gpuErrchk(cudaDeviceSynchronize());

			g_Seg.ClearSegments();

			gpuErrchk(cudaDeviceReset());

			return 0;
		}

		int PathIntegratorIncrementalAndStraight()
		{
			signal(SIGABRT, SignalHander);

			SetFilePtr(stdout);
			//SetBlockLog(1);

			Utility::FramePathTestConfig pathConfig;
			pathConfig.RefreshValues();
			Utility::TestProjectConfig projConfig;
			projConfig.RefreshValues();

			FILE* fp = fopen(projConfig.frameRequestPath, "rb");
			g_Seg.hreq = new FrameRequest();
			LoadFrameRequest(fp, &g_Seg.hreq, malloc);
			fclose(fp);

			OptimalLaunchParam param;
			GetOptimalBlockAndThreadDim(0, param);

			if (!pathConfig.imageConfigIgnore)
				pathConfig.Set(g_Seg.hreq->opt, param.threadCountinBlock, param.blockCountInGrid);

			g_Seg.hreq->opt.updateFunc = nullptr;
			g_Seg.hreq->opt.updateFrameFunc = update;

			g_Seg.target = new SimpleColorTarget(g_Seg.hreq->opt.resultImageResolution.x, g_Seg.hreq->opt.resultImageResolution.y);
			g_Seg.pi = new PathIntegrator(param, g_Seg.hreq->opt.resultImageResolution, (PathIntegratorPixelContent)pathConfig.pixelContent, true);

			{
				char dateStrPtr[257];
				time_t rawtime;
				struct tm timeinfo;
				time(&rawtime);
				localtime_s(&timeinfo, &rawtime);

				strftime(dateStrPtr, sizeof(dateStrPtr), "%Y-%m-%d_%H%M%S", &timeinfo);

				std::string requestPath = std::string(projConfig.frameRequestPath);

				int commaIndex = requestPath.find_last_of('.');
				std::string fileName;

				if (commaIndex == std::string::npos)
					fileName = requestPath.substr(requestPath.find_last_of('/') + 1);
				else
					fileName = requestPath.substr(requestPath.find_last_of('/') + 1, commaIndex - 1);

				sprintf(fileDirAndName, "./Images/%s_%s.ppm", fileName.c_str(), dateStrPtr);
			}

			AllocateDeviceFrameRequest(g_Seg.hreq, &g_Seg.din);

			g_Seg.threadCount = param.GetMaxThreadCount();
			g_Seg.aag = AcceleratedAggregate::GetAggregateDevice(&g_Seg.hreq->input, g_Seg.din, 0, g_Seg.threadCount);

			{
				TimeProfiler p = TimeProfiler("RenderStraight");
				g_Seg.pi->RenderStraight(GetAggregate, HostDevicePair<IMultipleInput*>(&g_Seg.hreq->input, g_Seg.din), *g_Seg.target, g_Seg.hreq->opt, param);
			}

			{
				TimeProfiler p = TimeProfiler("RenderIncremental");
				g_Seg.pi->RenderIncremental(GetAggregate, HostDevicePair<IMultipleInput*>(&g_Seg.hreq->input, g_Seg.din), *g_Seg.target, g_Seg.hreq->opt, param);
			}

			gpuErrchk(cudaDeviceSynchronize());

			g_Seg.ClearSegments();

			gpuErrchk(cudaDeviceReset());

			return 0;
		}

		void updateFrameFunc(int frameIndex, void* rgbaPtr)
		{
			char buffer[4096];
			sprintf(buffer, "./temp/%d", frameIndex);
			ImageWrite(buffer, ".png", (ColorRGBA*)rgbaPtr, g_Seg.hmreq->opt.resultImageResolution.x, g_Seg.hmreq->opt.resultImageResolution.y);

			if (frameIndex + 1 == g_Seg.hmreq->input.GetCount())
				VideoWrite("./temp/%s", ".png", "./temp/SimpleScene.avi", g_Seg.hmreq->opt.resultImageResolution.x, g_Seg.hmreq->opt.resultImageResolution.y, 60.0, g_Seg.hmreq->input.GetCount());
		}
		int PathIntegratorMultiFrameRequestProc()
		{
			signal(SIGABRT, SignalHander);

			SetFilePtr(stdout);
			//SetBlockLog(1);

			Utility::FramePathTestConfig pathConfig;
			pathConfig.RefreshValues();
			Utility::TestProjectConfig projConfig;
			projConfig.RefreshValues();

			FILE* fp = fopen(projConfig.multiFrameRequestPath, "rb");
			g_Seg.hmreq = new MultiFrameRequest();
			LoadMultiFrameRequest(fp, &g_Seg.hmreq, malloc);
			fclose(fp);

			OptimalLaunchParam param;
			GetOptimalBlockAndThreadDim(0, param);

			if (!pathConfig.imageConfigIgnore)
			{
				pathConfig.Set(g_Seg.hmreq->opt, param.threadCountinBlock, param.blockCountInGrid);
				g_Seg.hmreq->input.startIndex = pathConfig.startIndex;
				g_Seg.hmreq->input.endCount = pathConfig.endCount;
			}

			g_Seg.hmreq->opt.updateFunc = nullptr;
			g_Seg.hmreq->opt.updateFrameFunc = updateFrameFunc;

			g_Seg.target = new SimpleColorTarget(g_Seg.hmreq->opt.resultImageResolution.x, g_Seg.hmreq->opt.resultImageResolution.y);
			g_Seg.pi = new PathIntegrator(param, g_Seg.hmreq->opt.resultImageResolution, (PathIntegratorPixelContent)pathConfig.pixelContent, true);

			AllocateDeviceMultiFrameRequest(g_Seg.hmreq, &g_Seg.dmin);

			g_Seg.threadCount = param.GetMaxThreadCount();

			AcceleratedAggregate* deviceAAGArray = (AcceleratedAggregate*)MAllocDevice(sizeof(AcceleratedAggregate) * g_Seg.hmreq->input.in.mutableInputLen);
			AcceleratedAggregate::GetAggregateDevice(g_Seg.hmreq->input.in.mutableInputLen, deviceAAGArray, &g_Seg.hmreq->input, g_Seg.dmin, g_Seg.threadCount);
			g_Seg.aagArray = deviceAAGArray;

			{
				TimeProfiler p = TimeProfiler("RenderStraight");

				if (pathConfig.incrementalMode)
					g_Seg.pi->RenderIncremental(GetAggregate, HostDevicePair<IMultipleInput*>(&g_Seg.hmreq->input, g_Seg.dmin), *g_Seg.target, g_Seg.hmreq->opt, param);
				else
					g_Seg.pi->RenderStraight(GetAggregate, HostDevicePair<IMultipleInput*>(&g_Seg.hmreq->input, g_Seg.dmin), *g_Seg.target, g_Seg.hmreq->opt, param);
			}

			gpuErrchk(cudaDeviceSynchronize());

			g_Seg.ClearSegments();

			gpuErrchk(cudaDeviceReset());

			return 0;
		}
	}
}