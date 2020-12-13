#pragma once

extern "C"
{
	int __declspec(dllexport) __cdecl SaveSingleFrameRequest(const char* fileName, RadGrabber::FrameRequestMarshal* req);
	int __declspec(dllexport) __cdecl SaveMultiFrameRequest(const char* fileName, RadGrabber::MultiFrameRequestMarshal* req);

	int __declspec(dllexport) __cdecl GenerateSingleFrameIncremental(RadGrabber::FrameRequestMarshal* req);
	int __declspec(dllexport) __stdcall StopGenerateSingleFrame();
	int __declspec(dllexport) __stdcall IsSingleFrameGenerating();

	int __declspec(dllexport) __cdecl GenerateMultiFrameIncrementalRunitme(RadGrabber::MultiFrameRequest* req);
	int __declspec(dllexport) __cdecl GenerateMultiFrameIncremental(RadGrabber::MultiFrameRequestMarshal* req);
	int __declspec(dllexport) __cdecl GenerateMultiFrameIncrementalRanged(RadGrabber::MultiFrameRequestMarshal* req, int startIndex, int endCount);
	int __declspec(dllexport) __cdecl StopGenerateMultiFrame();
	int __declspec(dllexport) __cdecl IsMultiFrameGenerating();
	int __declspec(dllexport) __cdecl GetGeneratedCurrentFrameIndex();
}
