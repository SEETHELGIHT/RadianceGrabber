#include "Marshal.h"

#include "Pipeline.h"
#include "DeviceConfig.h"

#include "PTEntry.cuh"

using namespace RadGrabber;

FrameInput* deviceInput;

extern "C"
{
	/*
		TODO: asynchronous executed task
	*/
	int __declspec(dllexport) __stdcall GenerateSingleFrame(FrameRequest* req)
	{
		AllocateDeviceMem(req, &deviceInput);
		//IncrementalPTSampling(req, deviceInput);
		FreeDeviceMem(deviceInput);

		return 0;
	}
	int __declspec(dllexport) __stdcall GenerateSingleFrameIncremental(FrameRequest* req)
	{
		return 0;
	}
	int __declspec(dllexport) __stdcall StopAll()
	{
		return 0;
	}
}
