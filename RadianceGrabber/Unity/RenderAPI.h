#pragma once

#include "Unity/IUnityGraphics.h"

#include <stddef.h>

struct IUnityInterfaces;


// Super-simple "graphics abstraction". This is nothing like how a proper platform abstraction layer would look like;
// all this does is a base interface for whatever our plugin sample needs. Which is only "draw some triangles"
// and "modify a texture" at this point.
//
// There are implementations of this base class for D3D9, D3D11, OpenGL etc.; see individual RenderAPI_* files.
class RenderAPI
{
public:
	virtual ~RenderAPI() { }

	virtual void* BeginReadVertexBuffer(void* bufferHandle, int* bufferSize) = 0;
	virtual void EndReadVertexBuffer(void* bufferHandle) = 0;

	virtual void* BeginReadIndexBuffer(void* bufferHandle, int* bufferSize) = 0;
	virtual void EndReadIndexBuffer(void* bufferHandle) = 0;

	virtual void* BeginReadTexture2D(void* bufferHandle) = 0;
	virtual void EndReadTexture2D(void* bufferHandle) = 0;

	virtual void* BeginWriteTexture2D(void* bufferHandle) = 0;
	virtual void EndWriteTexture2D(void* bufferHandle) = 0;

	// Process general event like initialization, shutdown, device loss/reset etc.
	virtual void ProcessDeviceEvent(UnityGfxDeviceEventType type, IUnityInterfaces* interfaces) = 0;

public:
	static RenderAPI* GetRenderAPI();
	static void InitializeAPI(UnityGfxRenderer apiType);
	static void DestroyAPI();

private:
	static RenderAPI* mRenderAPI;
};


// Create a graphics API implementation instance for the given API type.
RenderAPI* CreateRenderAPI(UnityGfxRenderer apiType);

