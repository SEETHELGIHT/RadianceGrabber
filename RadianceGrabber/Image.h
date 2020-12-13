#include "DataTypes.cuh"

#pragma once

namespace RadGrabber
{
	int __declspec(dllexport) __cdecl ImageWrite(const char* path, const char* ext, RadGrabber::ColorRGBA* rgba, int width, int height);
	int __declspec(dllexport) __cdecl VideoWrite(const char* imagePathFormat, const char* imageExt, const char* videoPath, int width, int height, double fps, int frameCount);
	int __declspec(dllexport) __cdecl VideoWrite2(const char* imagePathFormat, const char* videoPath, double fps, int frameCount, int startIndex, int endCount);
}
