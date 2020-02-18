#include "Marshal.h"

#include "Pipeline.h"
#include "DeviceConfig.h"

#include "PTEntry.cuh"

using namespace RadGrabber;

UnityFrameInput* deviceInput;

extern "C"
{
	/*
		TODO: asynchronous executed task
	*/
	int __declspec(dllexport) __stdcall GenerateSingleFrame(UnityFrameRequest* req)
	{
		AllocateDeviceMem(req, &deviceInput);
		//IncrementalPTSampling(req, deviceInput);
		FreeDeviceMem(deviceInput);

		return 0;
	}
	int __declspec(dllexport) __stdcall GenerateSingleFrameIncremental(UnityFrameRequest* req)
	{
		return 0;
	}
	int __declspec(dllexport) __stdcall StopAll()
	{
		return 0;
	}
}
