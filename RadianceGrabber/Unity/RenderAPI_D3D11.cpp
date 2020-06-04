#include "RenderAPI.h"
#include "PlatformBase.h"

// Direct3D 11 implementation of RenderAPI.

#if SUPPORT_D3D11

#include <assert.h>
#include <d3d11.h>
#include "Unity/IUnityGraphicsD3D11.h"
#include "../Util.h"

class RenderAPI_D3D11 : public RenderAPI
{
public:
	RenderAPI_D3D11();
	virtual ~RenderAPI_D3D11() { }

	void* BeginReadVertexBuffer(void* bufferHandle, int* bufferSize);
	void EndReadVertexBuffer(void* bufferHandle);

	void* BeginReadIndexBuffer(void* bufferHandle, int* bufferSize);
	void EndReadIndexBuffer(void* bufferHandle);

	void* BeginReadTexture2D(void* bufferHandle);
	void EndReadTexture2D(void* bufferHandle);

	void* BeginWriteTexture2D(void* bufferHandle);
	void EndWriteTexture2D(void* bufferHandle);

	void ProcessDeviceEvent(UnityGfxDeviceEventType type, IUnityInterfaces* interfaces);

private:
	ID3D11Device* m_Device;
};

RenderAPI* CreateRenderAPI_D3D11()
{
	return new RenderAPI_D3D11();
}

RenderAPI_D3D11::RenderAPI_D3D11()
	: m_Device(NULL)
{
}

void RenderAPI_D3D11::ProcessDeviceEvent(UnityGfxDeviceEventType type, IUnityInterfaces* interfaces)
{
	switch (type)
	{
	case kUnityGfxDeviceEventInitialize:
	{
		IUnityGraphicsD3D11* d3d = interfaces->Get<IUnityGraphicsD3D11>();
		m_Device = d3d->GetDevice();
		break;
	}
	case kUnityGfxDeviceEventShutdown:
		break;
	}
}

void* RenderAPI_D3D11::BeginReadVertexBuffer(void* bufferHandle, int* bufferSize)
{
	ID3D11Buffer* d3dbuf = (ID3D11Buffer*)bufferHandle;
	assert(d3dbuf);
	D3D11_BUFFER_DESC desc;
	d3dbuf->GetDesc(&desc);
	*bufferSize = desc.ByteWidth;

	ID3D11DeviceContext* ctx = NULL;
	m_Device->GetImmediateContext(&ctx);
	D3D11_MAPPED_SUBRESOURCE mapped;
	ctx->Map(d3dbuf, 0, D3D11_MAP_READ, 0, &mapped);
	ctx->Release();

	return mapped.pData;
}


void RenderAPI_D3D11::EndReadVertexBuffer(void* bufferHandle)
{
	ID3D11Buffer* d3dbuf = (ID3D11Buffer*)bufferHandle;
	assert(d3dbuf);

	ID3D11DeviceContext* ctx = NULL;
	m_Device->GetImmediateContext(&ctx);
	ctx->Unmap(d3dbuf, 0);
	ctx->Release();
}

void * RenderAPI_D3D11::BeginReadIndexBuffer(void * bufferHandle, int * bufferSize)
{
	return BeginReadVertexBuffer(bufferHandle, bufferSize);
}

void RenderAPI_D3D11::EndReadIndexBuffer(void * bufferHandle)
{
	EndReadVertexBuffer(bufferHandle);
}

void * RenderAPI_D3D11::BeginReadTexture2D(void * bufferHandle)
{
	RadGrabber::Log("Begine::1\n");

	ID3D11Texture2D* d3dtex = (ID3D11Texture2D*)bufferHandle;
	assert(d3dtex);

	RadGrabber::Log("Begine::2\n");

	ID3D11DeviceContext* ctx = nullptr;
	m_Device->GetImmediateContext(&ctx);
	assert(ctx != nullptr);

	RadGrabber::Log("Begine::3\n");

	D3D11_MAPPED_SUBRESOURCE mapped;
	ctx->Map(d3dtex, 0, D3D11_MAP_READ, 0, &mapped);
	ctx->Release();

	RadGrabber::Log("Begine::last\n");

	return mapped.pData;
}

void RenderAPI_D3D11::EndReadTexture2D(void * bufferHandle)
{
	ID3D11Texture2D* d3dtex = (ID3D11Texture2D*)bufferHandle;
	assert(d3dtex);

	ID3D11DeviceContext* ctx = NULL;
	m_Device->GetImmediateContext(&ctx);
	ctx->Unmap(d3dtex, 0);
	ctx->Release();
}

void * RenderAPI_D3D11::BeginWriteTexture2D(void * bufferHandle)
{
	ID3D11Texture2D* d3dtex = (ID3D11Texture2D*)bufferHandle;
	assert(d3dtex);

	ID3D11DeviceContext* ctx = NULL;
	m_Device->GetImmediateContext(&ctx);
	D3D11_MAPPED_SUBRESOURCE mapped;
	ctx->Map(d3dtex, 0, D3D11_MAP_WRITE, 0, &mapped);
	ctx->Release();

	return mapped.pData;
}

void RenderAPI_D3D11::EndWriteTexture2D(void * bufferHandle)
{
	ID3D11Texture2D* d3dtex = (ID3D11Texture2D*)bufferHandle;
	assert(d3dtex);

	ID3D11DeviceContext* ctx = NULL;
	m_Device->GetImmediateContext(&ctx);
	ctx->Unmap(d3dtex, 0);
	ctx->Release();
}

#endif // #if SUPPORT_D3D11
