#include <Windows.h>
#include <thread>
#include <mutex>

#include "Marshal.cuh"

#include "Pipeline.h"
#include "DeviceConfig.h"
#include "ColorTarget.h"
#include "Util.h"

#include "PathIntegrator.h"
#include "IterativePathIntegrator.h"
#include "AcceleratedAggregate.h"
#include "LinearAggregate.h"

using namespace RadGrabber;

struct SingleFrameGenerationSegment
{
	std::thread* thread;
	void* threadNativeHandle;
	std::mutex clearSegmentMutex;

	FrameRequest* req;
	FrameInput* deviceInput;
	ICancelableIntergrator* integrator;
	IAggregate* aggregate;
	IColorTarget* target;
	bool startTerminate;
	bool onFrameGenerating;

	void ClearSegment()
	{
		SAFE_HOST_DELETE(integrator);
		SAFE_HOST_DELETE(target);
		
		if (aggregate)
		{
			AcceleratedAggregate::DestroyDeviceAggregate((AcceleratedAggregate*)aggregate);
			aggregate = nullptr;
		};
		if (deviceInput)
		{
			FreeDeviceFrameRequest(deviceInput);
			deviceInput = nullptr;
		}
		if (req)
		{
			delete req;
			req = nullptr;
		}
	}
};
SingleFrameGenerationSegment g_SingleFrameCalcSegment;
const IAggregate* GetConstAgg(int m) { return g_SingleFrameCalcSegment.aggregate; }

extern "C"
{
	/*
		TODO: thread pool
	*/

	int __declspec(dllexport) __cdecl SaveSingleFrameRequest(const char* fileName, RadGrabber::FrameRequestMarshal* req)
	{
		FrameRequest* other_req = new FrameRequest(*req);

		char buffer[4097];
		sprintf(buffer, "./%s.framerequest", fileName);

		FILE* fp = nullptr;
		fopen_s(&fp, buffer, "wb");
		Log("StoreFrameRequest return size::%ld\n", StoreFrameRequest(other_req, fp));
		fclose(fp);

		return 0;
	}
	int __declspec(dllexport) __cdecl SaveMultiFrameRequest(const char* fileName, RadGrabber::MultiFrameRequestMarshal* req)
	{
		MultiFrameRequest* other_req = new MultiFrameRequest(*req);

		char buffer[4097];
		sprintf(buffer, "./%s.multiframerequest", fileName);

		FILE* fp = nullptr;
		fopen_s(&fp, buffer, "wb");
		Log("StoreFrameRequest return size::%ld\n", StoreMultiFrameRequest(other_req, fp));
		fclose(fp);

		return 0;
	}

	void GenerateSingleFrameIncrementalInternal(void* ptr)
	{
		SingleFrameGenerationSegment* seg = &g_SingleFrameCalcSegment;
		seg->onFrameGenerating = true;

		Log("GenerateSingleFrameIncrementalInternal Start\n");

		AllocateDeviceFrameRequest(seg->req, &seg->deviceInput);

		int deviceID;
		cudaGetDevice(&deviceID);
		OptimalLaunchParam param;
		GetOptimalBlockAndThreadDim(deviceID, param);

		seg->integrator = new PathIntegrator(param, seg->req->opt.resultImageResolution);
		seg->aggregate = AcceleratedAggregate::GetAggregateDevice(&seg->req->input, seg->deviceInput, 0, param.GetMaxThreadCount());
		seg->target = new SingleFrameColorTarget(0 /*it must be configurated!*/, seg->req->opt.resultImageResolution, seg->req->output.pixelBuffer);

		seg->integrator->RenderIncremental(GetConstAgg, HostDevicePair<IMultipleInput*>(&seg->req->input, seg->deviceInput), *seg->target, seg->req->opt, param);

		Log("GenerateSingleFrameIncrementalInternal End\n");

		{
			std::unique_lock<std::mutex> lock(seg->clearSegmentMutex);

			if (!seg->startTerminate)
			{
				seg->thread = nullptr;
				seg->threadNativeHandle = nullptr;;
				seg->ClearSegment();
				seg->startTerminate = false;
				seg->onFrameGenerating = false;

				Log("GenerateSingleFrameIncrementalInternal Cleanup");
			}
			else
				Log("GenerateSingleFrameIncrementalInternal Passed as stopping thread cleanup");
		}
	}
	int __declspec(dllexport) __cdecl GenerateSingleFrameIncremental(RadGrabber::FrameRequestMarshal* req)
	{
		if (g_SingleFrameCalcSegment.req != nullptr)
			return -1;

		Log("GenerateSingleFrameIncremental :: Start");

		FrameRequest* other_req = new FrameRequest(*req);

		g_SingleFrameCalcSegment.req = other_req;
		g_SingleFrameCalcSegment.thread = new std::thread(GenerateSingleFrameIncrementalInternal, nullptr);
		g_SingleFrameCalcSegment.threadNativeHandle = g_SingleFrameCalcSegment.thread->native_handle();
		g_SingleFrameCalcSegment.thread->detach();

		Log("GenerateSingleFrameIncremental :: Delay");

		return 0;
	}
	int GenerateSingleFrameIncrementalTest(RadGrabber::FrameRequestMarshal* req, IColorTarget* target)
	{
		if (g_SingleFrameCalcSegment.req != nullptr)
			return -1;

		FrameRequest* other_req = new FrameRequest(*req);

		g_SingleFrameCalcSegment.req = other_req;
		g_SingleFrameCalcSegment.target = target;
		g_SingleFrameCalcSegment.thread = new std::thread(GenerateSingleFrameIncrementalInternal, &g_SingleFrameCalcSegment);
		g_SingleFrameCalcSegment.threadNativeHandle = g_SingleFrameCalcSegment.thread->native_handle();
		g_SingleFrameCalcSegment.thread->detach();

		return 0;
	}
	int StopGenerateSingleFrameInternal()
	{
		Log("StopGenerateSingleFrameInternal :: Start");

		{
			std::unique_lock<std::mutex> lock(g_SingleFrameCalcSegment.clearSegmentMutex);
			g_SingleFrameCalcSegment.startTerminate = true;
		}

		if (g_SingleFrameCalcSegment.thread)
		{
			if (g_SingleFrameCalcSegment.startTerminate)
			{
				Log("StopGenerateSingleFrameInternal Passed as execution thread cleanup");
				return 0;
			}
			
			g_SingleFrameCalcSegment.startTerminate = false;

			Log("StopGenerateSingleFrameInternal :: thread removing..");

			g_SingleFrameCalcSegment.integrator->ReserveCancel();

			while (!g_SingleFrameCalcSegment.integrator->IsCancel());

			Log("StopGenerateSingleFrameInternal :: thread exited!");

			TerminateThread(g_SingleFrameCalcSegment.threadNativeHandle, 0);
			WaitForSingleObject(g_SingleFrameCalcSegment.threadNativeHandle, INFINITE);

			delete g_SingleFrameCalcSegment.thread;
			g_SingleFrameCalcSegment.thread = nullptr;
			g_SingleFrameCalcSegment.threadNativeHandle = nullptr;

			g_SingleFrameCalcSegment.ClearSegment();
			g_SingleFrameCalcSegment.onFrameGenerating = false;
		}

		Log("StopGenerateSingleFrameInternal :: End");

		return 0;
	}
	int __declspec(dllexport) __cdecl StopGenerateSingleFrame()
	{
		if (g_SingleFrameCalcSegment.onFrameGenerating && !g_SingleFrameCalcSegment.startTerminate)
			new std::thread(StopGenerateSingleFrameInternal);
		else
			return -1;

		return 0;
	}
	int __declspec(dllexport) __cdecl IsSingleFrameGenerating()
	{
		return g_SingleFrameCalcSegment.onFrameGenerating;
	}

	struct MultiFrameGenerationSegment
	{
		std::thread* thread;
		void* threadNativeHandle;
		std::mutex clearSegmentMutex;

		MultiFrameRequest* req;
		MultiFrameInput* deviceInput;
		ICancelableIntergrator* integrator;
		AcceleratedAggregate* aggregateArray;
		IColorTarget* target;
		bool startTerminate;
		bool onFrameGenerating;

		void ClearSegment()
		{
			SAFE_HOST_DELETE(integrator);
			SAFE_HOST_DELETE(target);

			if (aggregateArray)
			{
				AcceleratedAggregate::DestroyDeviceAggregate(req->input.in.mutableInputLen, aggregateArray);
				aggregateArray = nullptr;
			};
			if (deviceInput)
			{
				FreeDeviceMultiFrameRequest(deviceInput);
				deviceInput = nullptr;
			}
			if (req)
			{
				delete req;
				req = nullptr;
			}
		}

		void AfterImage(int frameCount)
		{
		}
	};
	MultiFrameGenerationSegment g_MultiFrameCalcSegment;
	const IAggregate* GetAggregate(int m) { return g_MultiFrameCalcSegment.aggregateArray + m; }

	int __declspec(dllexport) __cdecl GenerateMultiFrame(MultiFrameRequest* req)
	{
		return 0;
	}

	void GenerateMultiFrameIncrementalInternal(MultiFrameGenerationSegment* seg)
	{
		seg->onFrameGenerating = true;

		Log("GenerateMultiFrameIncrementalInternal Start\n");

		AllocateDeviceMultiFrameRequest(seg->req, &seg->deviceInput);

		OptimalLaunchParam param;
		GetOptimalBlockAndThreadDim(0, param);

		AcceleratedAggregate* deviceAAGArray = (AcceleratedAggregate*)MAllocDevice(sizeof(AcceleratedAggregate) * seg->req->input.in.mutableInputLen);
		AcceleratedAggregate::GetAggregateDevice(seg->req->input.GetCount(), deviceAAGArray, &seg->req->input, seg->deviceInput, param.GetMaxThreadCount());
		seg->aggregateArray = deviceAAGArray;

		seg->integrator = new PathIntegrator(param, seg->req->opt.resultImageResolution);
		//seg->aggregate = AcceleratedAggregate::GetAggregateDevice(&seg->req->input, seg->deviceInput, 0, param.GetMaxThreadCount());
		seg->target = new SingleFrameColorTarget(0 /*it must be configurated!*/, seg->req->opt.resultImageResolution, seg->req->output.pixelBuffer);

		seg->integrator->RenderStraight(GetAggregate, HostDevicePair<IMultipleInput*>(&seg->req->input, seg->deviceInput), *seg->target, seg->req->opt, param);

		Log("GenerateMultiFrameIncrementalInternal End\n");

		{
			std::unique_lock<std::mutex> lock(seg->clearSegmentMutex);

			if (!seg->startTerminate)
			{
				seg->thread = nullptr;
				seg->threadNativeHandle = nullptr;;
				seg->ClearSegment();
				seg->startTerminate = false;
				seg->onFrameGenerating = false;

				Log("GenerateMultiFrameIncrementalInternal Cleanup");
			}
			else
				Log("GenerateMultiFrameIncrementalInternal Passed as stopping thread cleanup");
		}
	}
	int __declspec(dllexport) __cdecl GenerateMultiFrameIncremental(MultiFrameRequestMarshal* req)
	{
		if (g_MultiFrameCalcSegment.req != nullptr)
			return -1;

		MultiFrameRequest* other_req = new MultiFrameRequest(*req);

		g_MultiFrameCalcSegment.req = other_req;
		g_MultiFrameCalcSegment.thread = new std::thread(GenerateMultiFrameIncrementalInternal, &g_MultiFrameCalcSegment);
		g_MultiFrameCalcSegment.threadNativeHandle = g_MultiFrameCalcSegment.thread->native_handle();
		g_MultiFrameCalcSegment.thread->detach();

		return 0;
	}
	int __declspec(dllexport) __cdecl GenerateMultiFrameIncrementalRunitme(MultiFrameRequest* req)
	{
		if (g_MultiFrameCalcSegment.req != nullptr)
			return -1;

		g_MultiFrameCalcSegment.req = req;
		g_MultiFrameCalcSegment.thread = new std::thread(GenerateMultiFrameIncrementalInternal, &g_MultiFrameCalcSegment);
		g_MultiFrameCalcSegment.threadNativeHandle = g_MultiFrameCalcSegment.thread->native_handle();
		g_MultiFrameCalcSegment.thread->detach();

		return 0;
	}
	int __declspec(dllexport) __cdecl GenerateMultiFrameIncrementalRanged(RadGrabber::MultiFrameRequestMarshal* req, int startIndex, int endCount)
	{
		if (g_MultiFrameCalcSegment.req != nullptr)
			return -1;

		MultiFrameRequest* other_req = new MultiFrameRequest(*req, startIndex, endCount);

		g_MultiFrameCalcSegment.req = other_req;
		g_MultiFrameCalcSegment.thread = new std::thread(GenerateMultiFrameIncrementalInternal, &g_MultiFrameCalcSegment);
		g_MultiFrameCalcSegment.threadNativeHandle = g_MultiFrameCalcSegment.thread->native_handle();
		g_MultiFrameCalcSegment.thread->detach();

		return 0;
	}
	int StopGenerateMultiFrameInternal()
	{
		Log("StopGenerateMultiFrameInternal :: Start");

		{
			std::unique_lock<std::mutex> lock(g_MultiFrameCalcSegment.clearSegmentMutex);
			g_MultiFrameCalcSegment.startTerminate = true;
		}

		if (g_MultiFrameCalcSegment.thread)
		{
			if (g_MultiFrameCalcSegment.startTerminate)
			{
				Log("StopGenerateMultiFrameInternal Passed as execution thread cleanup");
				return 0;
			}

			g_MultiFrameCalcSegment.startTerminate = false;

			Log("StopGenerateMultiFrameInternal :: thread removing..");

			g_MultiFrameCalcSegment.integrator->ReserveCancel();

			while (!g_MultiFrameCalcSegment.integrator->IsCancel());

			TerminateThread(g_MultiFrameCalcSegment.threadNativeHandle, 0);
			WaitForSingleObject(g_MultiFrameCalcSegment.threadNativeHandle, INFINITE);

			Log("StopGenerateMultiFrameInternal :: thread exited!");

			if (g_MultiFrameCalcSegment.thread)
			{
				delete g_MultiFrameCalcSegment.thread;
				g_MultiFrameCalcSegment.thread = nullptr;
				g_MultiFrameCalcSegment.threadNativeHandle = nullptr;
			}

			g_MultiFrameCalcSegment.ClearSegment();
			g_MultiFrameCalcSegment.onFrameGenerating = false;
		}

		Log("StopGenerateMultiFrameInternal :: End");

		return 0;
	}
	int __declspec(dllexport) __cdecl StopGenerateMultiFrame()
	{
		if (g_MultiFrameCalcSegment.onFrameGenerating && !g_MultiFrameCalcSegment.startTerminate)
			new std::thread(StopGenerateMultiFrameInternal);
		else
			return -1;

		return 0;
	}
	int __declspec(dllexport) __cdecl IsMultiFrameGenerating()
	{
		return g_MultiFrameCalcSegment.onFrameGenerating;
	}
	int __declspec(dllexport) __cdecl GetGeneratedCurrentFrameIndex()
	{
		return 0;
	}
}
