#pragma once

class RadGrabber::FrameRequest;
class RadGrabber::IColorTarget;

extern "C"
{
	int __declspec(dllexport) __cdecl GenerateSingleFrame(RadGrabber::FrameRequest* req);
	int __declspec(dllexport) __cdecl GenerateSingleFrameIncremental(RadGrabber::FrameRequestMarshal* req);
	int __declspec(dllexport) __cdecl SaveSingleFrameRequest(const char* fileName, RadGrabber::FrameRequestMarshal* req);
	int __declspec(dllexport) __stdcall StopGenerateSingleFrame();
	int __declspec(dllexport) __stdcall IsSingleFrameGenerating();

	int __declspec(dllexport) __stdcall GenerateTest(RadGrabber::FrameRequest* req);

	int GenerateSingleFrameIncrementalTest(RadGrabber::FrameRequestMarshal* req, RadGrabber::IColorTarget* target);
}
