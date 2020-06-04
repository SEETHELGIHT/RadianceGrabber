#include <Windows.h>
#include <thread>
#include <mutex>

#include "Marshal.cuh"

#include "Pipeline.h"
#include "DeviceConfig.h"
#include "Integrator.h"
#include "Aggregate.h"
#include "ColorTarget.h"
#include "Util.h"

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

	void ClearSegment()
	{
		if (integrator)
		{
			delete integrator;
			integrator = nullptr;
		}
		if (target)
		{
			delete target;
			target = nullptr;
		}
		if (aggregate)
		{
			LinearAggregate::DestroyDeviceAggregate((LinearAggregate*)aggregate);
			aggregate = nullptr;
		};
		Log("ClearSegment::4\n");
		if (deviceInput)
		{
			FreeDeviceMem(deviceInput);
			deviceInput = nullptr;
		}
		Log("ClearSegment::5\n");

		if (req)
		{
			delete req;
			req = nullptr;
		}
		Log("ClearSegment::7\n");
	}
};
SingleFrameGenerationSegment g_SingleFrameCalcSegment;

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

	void GenerateSingleFrameIncrementalInternal(SingleFrameGenerationSegment* seg)
	{
		while (!g_SingleFrameCalcSegment.threadNativeHandle);

		Log("GenerateSingleFrameIncrementalInternal Start\n");

		AllocateDeviceMem(seg->req, &seg->deviceInput);

		OptimalLaunchParam param;
		GetOptimalBlockAndThreadDim(0, param);

		seg->integrator = new PathIntegrator(seg->req->opt);
		seg->aggregate = AcceleratedAggregate::GetAggregateDevice(&seg->req->input, seg->deviceInput, 0, param.threadCountinBlock.x * param.blockCountInGrid.x);
		seg->target = new SingleFrameColorTarget(0 /*it must be configurated!*/, seg->req->opt.resultImageResolution, seg->req->output.pixelBuffer);

		seg->integrator->Render(*seg->aggregate, seg->req->input, *seg->deviceInput, seg->req->opt, *seg->target, param);

		Log("GenerateSingleFrameIncrementalInternal End\n");

		std::lock_guard<std::mutex> lock(seg->clearSegmentMutex);

		if (seg->thread)
		{
			seg->thread = nullptr;
			seg->threadNativeHandle = nullptr;;
			seg->ClearSegment();
			seg->startTerminate = false;

			Log("GenerateSingleFrameIncrementalInternal Cleanup");
		}
	}
	int __declspec(dllexport) __cdecl GenerateSingleFrameIncremental(RadGrabber::FrameRequestMarshal* req)
	{
		if (g_SingleFrameCalcSegment.req != nullptr)
			return -1;

		FrameRequest* other_req = new FrameRequest(*req);

		g_SingleFrameCalcSegment.req = other_req;
		g_SingleFrameCalcSegment.thread = new std::thread(GenerateSingleFrameIncrementalInternal, &g_SingleFrameCalcSegment);
		g_SingleFrameCalcSegment.threadNativeHandle = g_SingleFrameCalcSegment.thread->native_handle();
		g_SingleFrameCalcSegment.thread->detach();

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
		std::lock_guard<std::mutex> lock(g_SingleFrameCalcSegment.clearSegmentMutex);

		if (g_SingleFrameCalcSegment.thread)
		{
			g_SingleFrameCalcSegment.startTerminate = true;

			g_SingleFrameCalcSegment.integrator->ReserveCancel();

			while (!g_SingleFrameCalcSegment.integrator->IsCancel());

			TerminateThread(g_SingleFrameCalcSegment.threadNativeHandle, 0);
			WaitForSingleObject(g_SingleFrameCalcSegment.threadNativeHandle, INFINITE);

			if (g_SingleFrameCalcSegment.thread)
			{
				delete g_SingleFrameCalcSegment.thread;
				g_SingleFrameCalcSegment.thread = nullptr;
				g_SingleFrameCalcSegment.threadNativeHandle = nullptr;
			}

			g_SingleFrameCalcSegment.ClearSegment();	
			g_SingleFrameCalcSegment.startTerminate = false;
		}

		return 0;
	}
	int __declspec(dllexport) __stdcall StopGenerateSingleFrame()
	{
		if (g_SingleFrameCalcSegment.thread)
			new std::thread(StopGenerateSingleFrameInternal);
		else
			return -1;

		return 0;
	}
	int __declspec(dllexport) __stdcall IsSingleFrameGenerating()
	{
		return g_SingleFrameCalcSegment.thread != nullptr;
	}

	int __declspec(dllexport) __stdcall GenerateMultiFrame(MultiFrameRequest* req)
	{
		return 0;
	}
	int __declspec(dllexport) __stdcall GenerateMultiFrameIncremental(MultiFrameRequest* req)
	{
		return 0;
	}
	int StopGenerateMultiFrame()
	{
		return 0;
	}
	int GetGeneratedCurrentFrameIndex()
	{
		return 0;
	}

	int __declspec(dllexport) __stdcall StopAll()
	{
		return 0;
	}
}
