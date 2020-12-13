#include "Pipeline.h"

#include "Marshal.cuh"

/*
	TODO:: using custom memory allocator 
	ASSERT_IS_FALSE\(cudaMalloc\(\&*(.+)\,\ 
	$1 = MAllocDevice(
*/
namespace RadGrabber
{
#pragma region DECLARE_SUBROUTINES

	__host__ FrameImmutableInput AllocateDeviceImmutableData(FrameImmutableInput* hostIn, FrameImmutableInput* deviceIn);
	__host__ FrameMutableInput AllocateDeviceMutableData(FrameMutableInput* hostIn, FrameMutableInput* deviceIn);
	__host__ void FreeDeviceImmutableData(FrameImmutableInput* deviceInputInHost);
	__host__ void FreeDeviceMutableData(FrameMutableInput* deviceInputInHost);
	__host__ void FreeHostMutableData(FrameMutableInput* hostIn);
	__host__ void FreeHostImmutableData(FrameImmutableInput* hostIn);

	__host__ size_t StoreMeshChunk(MeshChunk* m, FILE* fp);
	__host__ size_t StoreTexture2DChunk(Texture2DChunk* t, FILE* fp);
	__host__ size_t StoreMeshRendererChunk(MeshRendererChunk* mr, FILE* fp);
	__host__ size_t StoreSkinnedMeshRendererChunk(SkinnedMeshRendererChunk* smr, FILE* fp);
	__host__ size_t StoreMutableData(FrameMutableInput* hostIn, FILE* fp);
	__host__ size_t StoreImmutableData(FrameImmutableInput* hostIn, FILE* fp);

	__host__ void LoadMeshChunk(MeshChunk* m, FILE* fp, void* (*allocator)(size_t cb));
	__host__ void LoadTexture2DChunk(Texture2DChunk* t, FILE* fp, void* (*allocator)(size_t cb));
	__host__ void LoadMeshRendererChunk(MeshRendererChunk* mr, FILE* fp, void* (*allocator)(size_t cb));
	__host__ void LoadSkinnedMeshRendererChunk(SkinnedMeshRendererChunk* smr, FILE* fp, void* (*allocator)(size_t cb));
	__host__ void LoadMutableData(FrameMutableInput* hostIn, FILE* fp, void* (*allocator)(size_t cb));
	__host__ void LoadImmutableData(FrameImmutableInput* hostIn, FILE* fp, void* (*allocator)(size_t cb));

#pragma endregion DECLARE_SUBROUTINES

#pragma region IMPLEMENT_REQUEST_FUNCTIONS

	__global__ void SetDeviceFrameInput(FrameInput* din, FrameInputInternal in)
	{
		if (threadIdx.x == 0)
			din = new (din)FrameInput(in);
	}

	__host__ void AllocateDeviceFrameRequest(FrameRequest* hostReq, FrameInput** outDeviceInput)
	{
		ASSERT(outDeviceInput);

		FrameInput hin = hostReq->input;
		FrameInputInternal dib = hostReq->input.in;

		dib.mutableInput = AllocateDeviceMutableData(&hin.in.mutableInput, nullptr);
		dib.immutableInput = AllocateDeviceImmutableData(&hin.in.immutableInput, nullptr);

		gpuErrchk(cudaMalloc(outDeviceInput, sizeof(FrameInput)));
		SetDeviceFrameInput << <1, 1 >> > (*outDeviceInput, dib);
	}
	__host__ void FreeHostFrameRequest(FrameRequest* hreq)
	{
		FreeHostMutableData(&hreq->input.in.mutableInput);
		FreeHostImmutableData(&hreq->input.in.immutableInput);

		SAFE_HOST_FREE(hreq);
	}
	__host__ void FreeDeviceFrameRequest(FrameInput* deviceInput)
	{
		FrameInput in;
		gpuErrchk(cudaMemcpy(&in, deviceInput, sizeof(FrameInput), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		FreeDeviceMutableData(&in.in.mutableInput);
		FreeDeviceImmutableData(&in.in.immutableInput);

		gpuErrchk(cudaFree(deviceInput));
	}

	__host__ size_t StoreFrameRequest(FrameRequest* hostReq, FILE* fp)
	{
		size_t size = 0;
		{
			Log("%x, %x", hostReq->opt.updateFunc, hostReq->opt.updateFrameFunc);
			size += fwrite(&hostReq->opt, sizeof(RequestOption), 1, fp);
		}

		{
			FrameInputInternal& in = hostReq->input.in;

			size += StoreImmutableData(&in.immutableInput, fp);
			size += StoreMutableData(&in.mutableInput, fp);
		}

		{
			size += fwrite(&hostReq->output, sizeof(FrameOutput), 1, fp);
		}

		fflush(fp);
		FlushLog();

		return size;
	}
	__host__ void LoadFrameRequest(FILE* fp, FrameRequest** reqBuffer, void* (*allocator)(size_t cb))
	{
		{
			fread(&(*reqBuffer)->opt, sizeof(RequestOption), 1, fp);
		}

		{
			FrameInput& in = (*reqBuffer)->input;

			LoadImmutableData(&in.in.immutableInput, fp, allocator);
			LoadMutableData(&in.in.mutableInput, fp, allocator);
		}

		{
			fread(&(*reqBuffer)->output, sizeof(FrameOutput), 1, fp);
		}
	}

	__global__ void SetDeviceMultiFrameInput(MultiFrameInput* din, MultiFrameInputInternal in, int startIndex, int endCount)
	{
		if (threadIdx.x == 0)
		{
			din = new (din)MultiFrameInput(in);
			din->startIndex = startIndex;
			din->endCount = endCount;
		}			
	}

	__host__ void AllocateDeviceMultiFrameRequest(MultiFrameRequest* hostReq, MultiFrameInput** outDeviceInput)
	{
		ASSERT(outDeviceInput);

		MultiFrameInput hin = hostReq->input;
		MultiFrameInputInternal dib = hostReq->input.in;

		dib.immutable = AllocateDeviceImmutableData(&hin.in.immutable, nullptr);
		dib.mutableInputs = (FrameMutableInput*)MAllocDevice(sizeof(FrameMutableInput) * dib.mutableInputLen);
		for (int i = 0; i < dib.mutableInputLen; i++)
		{
			FrameMutableInput min = AllocateDeviceMutableData(hin.in.mutableInputs + i, nullptr);;
			gpuErrchk(cudaMemcpy(dib.mutableInputs + i, &min, sizeof(FrameMutableInput), cudaMemcpyKind::cudaMemcpyHostToDevice));
		}

		gpuErrchk(cudaMalloc(outDeviceInput, sizeof(FrameInput)));
		SetDeviceMultiFrameInput << <1, 1 >> > (*outDeviceInput, dib, hin.startIndex, hin.endCount);
	}
	__host__ void FreeDeviceMultiFrameRequest(MultiFrameInput* deviceInput)
	{
		MultiFrameInput in;
		gpuErrchk(cudaMemcpy(&in, deviceInput, sizeof(MultiFrameInput), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		FreeDeviceImmutableData(&in.in.immutable);
		for (int i = 0; i < in.in.mutableInputLen; i++)
		{
			FrameMutableInput cin;
			gpuErrchk(cudaMemcpy(&cin, in.in.mutableInputs + i, sizeof(FrameMutableInput), cudaMemcpyKind::cudaMemcpyDeviceToHost));
			FreeDeviceMutableData(&cin);
		}
			
		SAFE_DEVICE_DELETE(in.in.mutableInputs);
		SAFE_DEVICE_DELETE(deviceInput);
	}
	__host__ void FreeHostMultiFrameRequest(MultiFrameRequest* hreq)
	{
		FreeHostImmutableData(&hreq->input.in.immutable);
		for (int i = 0; i < hreq->input.in.mutableInputLen; i++)
			FreeHostMutableData(&hreq->input.in.mutableInputs[i]);
		SAFE_HOST_FREE(hreq->input.in.mutableInputs);

		SAFE_HOST_FREE(hreq);
	}
	__host__ size_t StoreMultiFrameRequest(MultiFrameRequest* hostReq, FILE* fp)
	{
		size_t size = 0;
		{
			size += fwrite(&hostReq->opt, sizeof(RequestOption), 1, fp);
		}

		{
			MultiFrameInputInternal& in = hostReq->input.in;

			size += StoreImmutableData(&in.immutable, fp);
			size += fwrite(&in.mutableInputLen, sizeof(int), 1, fp);
			for (int i = 0; i < in.mutableInputLen; i++)
				size += StoreMutableData(in.mutableInputs + i, fp);
		}

		{
			size += fwrite(&hostReq->output, sizeof(FrameOutput), 1, fp);
		}

		fflush(fp);
		FlushLog();

		return size;
	}
	__host__ void LoadMultiFrameRequest(FILE* fp, MultiFrameRequest** reqBuffer, void* (*allocator)(size_t cb))
	{
		{
			fread(&(*reqBuffer)->opt, sizeof(RequestOption), 1, fp);
		}

		{
			MultiFrameInput& in = (*reqBuffer)->input;

			LoadImmutableData(&in.in.immutable, fp, allocator);
			fread(&in.in.mutableInputLen, sizeof(int), 1, fp);
			in.in.mutableInputs = (FrameMutableInput*)allocator(sizeof(FrameMutableInput) * in.in.mutableInputLen);
			for (int i = 0; i < in.in.mutableInputLen; i++)
				LoadMutableData(in.in.mutableInputs + i, fp, allocator);
		}

		{
			fread(&(*reqBuffer)->output, sizeof(MultiFrameOutput), 1, fp);
		}
	}

#pragma endregion IMPLEMENT_REQUEST_FUNCTIONS

#pragma region IMPLEMENT_SUBROUTINES

	__host__ FrameImmutableInput AllocateDeviceImmutableData(FrameImmutableInput* hostIn, FrameImmutableInput* deviceIn)
	{
		FrameImmutableInput hin = *hostIn;
		FrameImmutableInput din = *hostIn;

		gpuErrchk(cudaMalloc(&din.meshBuffer, sizeof(MeshChunk) * hin.meshBufferLen));
		for (int i = 0; i < hin.meshBufferLen; i++)
		{
			MeshChunk c = hin.meshBuffer[i], c2 = c;

			gpuErrchk(cudaMalloc(&c.positions, sizeof(Vector3f) * c.vertexCount));
			gpuErrchk(cudaMemcpy(c.positions, c2.positions, sizeof(Vector3f) * c.vertexCount, cudaMemcpyKind::cudaMemcpyHostToDevice));
			gpuErrchk(cudaMalloc(&c.normals, sizeof(Vector3f) * c.vertexCount));
			gpuErrchk(cudaMemcpy(c.normals, c2.normals, sizeof(Vector3f) * c.vertexCount, cudaMemcpyKind::cudaMemcpyHostToDevice));
			gpuErrchk(cudaMalloc(&c.tangents, sizeof(Vector4f) * c.vertexCount));
			gpuErrchk(cudaMemcpy(c.tangents, c2.tangents, sizeof(Vector4f) * c.vertexCount, cudaMemcpyKind::cudaMemcpyHostToDevice));
			if (c2.uvs)
			{
				gpuErrchk(cudaMalloc(&c.uvs, sizeof(Vector2f) * c.vertexCount));
				gpuErrchk(cudaMemcpy(c.uvs, c2.uvs, sizeof(Vector2f) * c.vertexCount, cudaMemcpyKind::cudaMemcpyHostToDevice));
			}
			else
				c.uvs = nullptr;
			gpuErrchk(cudaMalloc(&c.indices, sizeof(int) * c.indexCount));
			gpuErrchk(cudaMemcpy(c.indices, c2.indices, sizeof(int) * c.indexCount, cudaMemcpyKind::cudaMemcpyHostToDevice));
			gpuErrchk(cudaMalloc(&c.submeshArrayPtr, sizeof(UnitySubMeshDescriptor) * c.submeshCount));
			gpuErrchk(cudaMemcpy(c.submeshArrayPtr, c2.submeshArrayPtr, sizeof(UnitySubMeshDescriptor) * c.submeshCount, cudaMemcpyKind::cudaMemcpyHostToDevice));

			if (c.bindposeCount > 0)
			{
				gpuErrchk(cudaMalloc(&c.bindposeArrayPtr, sizeof(Matrix4x4) * c.bindposeCount));
				gpuErrchk(cudaMemcpy(c.bindposeArrayPtr, c2.bindposeArrayPtr, sizeof(Matrix4x4) * c.bindposeCount, cudaMemcpyKind::cudaMemcpyHostToDevice));
			}
			else
				c.bindposeArrayPtr = nullptr;

			gpuErrchk(cudaMemcpy(din.meshBuffer + i, &c, sizeof(MeshChunk), cudaMemcpyKind::cudaMemcpyHostToDevice));
		}

		gpuErrchk(cudaMalloc(&din.skinnedMeshBuffer, sizeof(MeshChunk) * hin.skinnedMeshBufferLen));
		for (int i = 0; i < hin.skinnedMeshBufferLen; i++)
		{
			MeshChunk c = hin.skinnedMeshBuffer[i], c2 = c;

			gpuErrchk(cudaMalloc(&c.positions, sizeof(Vector3f) * c.vertexCount));
			gpuErrchk(cudaMemcpy(c.positions, c2.positions, sizeof(Vector3f) * c.vertexCount, cudaMemcpyKind::cudaMemcpyHostToDevice));
			gpuErrchk(cudaMalloc(&c.normals, sizeof(Vector3f) * c.vertexCount));
			gpuErrchk(cudaMemcpy(c.normals, c2.normals, sizeof(Vector3f) * c.vertexCount, cudaMemcpyKind::cudaMemcpyHostToDevice));
			gpuErrchk(cudaMalloc(&c.uvs, sizeof(Vector2f) * c.vertexCount));
			gpuErrchk(cudaMemcpy(c.uvs, c2.uvs, sizeof(Vector2f) * c.vertexCount, cudaMemcpyKind::cudaMemcpyHostToDevice));
			gpuErrchk(cudaMalloc(&c.indices, sizeof(int) * c.indexCount));
			gpuErrchk(cudaMemcpy(c.indices, c2.indices, sizeof(int) * c.indexCount, cudaMemcpyKind::cudaMemcpyHostToDevice));
			gpuErrchk(cudaMalloc(&c.submeshArrayPtr, sizeof(UnitySubMeshDescriptor) * c.submeshCount));
			gpuErrchk(cudaMemcpy(c.submeshArrayPtr, c2.submeshArrayPtr, sizeof(UnitySubMeshDescriptor) * c.submeshCount, cudaMemcpyKind::cudaMemcpyHostToDevice));

			if (c.bindposeCount > 0)
			{
				gpuErrchk(cudaMalloc(&c.bindposeArrayPtr, sizeof(Matrix4x4) * c.bindposeCount));
				gpuErrchk(cudaMemcpy(c.bindposeArrayPtr, c2.bindposeArrayPtr, sizeof(Matrix4x4) * c.bindposeCount, cudaMemcpyKind::cudaMemcpyHostToDevice));
			}
			else
				c.bindposeArrayPtr = nullptr;

			gpuErrchk(cudaMemcpy(din.skinnedMeshBuffer + i, &c, sizeof(MeshChunk), cudaMemcpyKind::cudaMemcpyHostToDevice));
		}

		gpuErrchk(cudaMalloc(&din.textureBuffer, sizeof(Texture2DChunk) * hin.textureBufferLen));
		for (int i = 0; i < hin.textureBufferLen; i++)
		{
			Texture2DChunk c = hin.textureBuffer[i], c2 = c;

			cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKind::cudaChannelFormatKindUnsigned);
			cudaArray* pixelArray = nullptr;
			gpuErrchk(cudaMallocArray(&pixelArray, &channelDesc, c.size.x, c.size.y));
			gpuErrchk(cudaMemcpyToArray(pixelArray, 0, 0, c2.pixelPtr, sizeof(ColorRGBA32) * c.size.x * c.size.y, cudaMemcpyKind::cudaMemcpyHostToDevice));

			/*
				SHOULD TODO ::
				resource 와 texture 자원을 분리하여 관리
			*/
			struct cudaResourceDesc resDesc;
			memset(&resDesc, 0, sizeof(struct cudaResourceDesc));
			resDesc.resType = cudaResourceTypeArray;
			resDesc.res.array.array = pixelArray;

			struct cudaTextureDesc texDesc;
			memset(&texDesc, 0, sizeof(struct cudaTextureDesc));
			texDesc.addressMode[0] = cudaAddressModeWrap;
			texDesc.addressMode[1] = cudaAddressModeWrap;
			texDesc.filterMode = (cudaTextureFilterMode)!!(int)c.filter;
			texDesc.readMode = cudaReadModeNormalizedFloat;
			texDesc.normalizedCoords = 1;

			ASSERT(sizeof(cudaTextureObject_t) == sizeof(void*));
			cudaCreateTextureObject((cudaTextureObject_t*)&c.pixelPtr, &resDesc, &texDesc, nullptr);

			gpuErrchk(cudaMemcpy(din.textureBuffer + i, &c, sizeof(Texture2DChunk), cudaMemcpyKind::cudaMemcpyHostToDevice));
		}

		if (deviceIn != nullptr)
			gpuErrchk(cudaMemcpy(deviceIn, &din, sizeof(FrameImmutableInput), cudaMemcpyKind::cudaMemcpyHostToDevice));

		return din;
	}
	__host__ FrameMutableInput AllocateDeviceMutableData(FrameMutableInput* hostIn, FrameMutableInput* deviceIn)
	{
		FrameMutableInput hin = *hostIn;
		FrameMutableInput din = *hostIn;

		gpuErrchk(cudaMalloc(&din.cameraBuffer, sizeof(CameraChunk) * hin.cameraBufferLen));
		gpuErrchk(cudaMemcpy(din.cameraBuffer, hin.cameraBuffer, sizeof(CameraChunk) * hin.cameraBufferLen, cudaMemcpyKind::cudaMemcpyHostToDevice));
		gpuErrchk(cudaMalloc(&din.lightBuffer, sizeof(LightChunk) * hin.lightBufferLen));
		gpuErrchk(cudaMemcpy(din.lightBuffer, hin.lightBuffer, sizeof(LightChunk) * hin.lightBufferLen, cudaMemcpyKind::cudaMemcpyHostToDevice));
		gpuErrchk(cudaMalloc(&din.materialBuffer, sizeof(MaterialChunk) * hin.materialBufferLen));
		gpuErrchk(cudaMemcpy(din.materialBuffer, hin.materialBuffer, sizeof(MaterialChunk) * hin.materialBufferLen, cudaMemcpyKind::cudaMemcpyHostToDevice));
		gpuErrchk(cudaMalloc(&din.skyboxMaterialBuffer, sizeof(SkyboxChunk) * hin.skyboxMaterialBufferLen));
		gpuErrchk(cudaMemcpy(din.skyboxMaterialBuffer, hin.skyboxMaterialBuffer, sizeof(SkyboxChunk) * hin.skyboxMaterialBufferLen, cudaMemcpyKind::cudaMemcpyHostToDevice));

		gpuErrchk(cudaMalloc(&din.meshRendererBuffer, sizeof(MeshRendererChunk) * hin.meshRendererBufferLen));
		for (int i = 0; i < hin.meshRendererBufferLen; i++)
		{
			MeshRendererChunk c = hin.meshRendererBuffer[i], c2 = c;

			gpuErrchk(cudaMalloc(&c.materialArrayPtr, sizeof(int) * c.materialCount));
			gpuErrchk(cudaMemcpy(c.materialArrayPtr, c2.materialArrayPtr, sizeof(int) * c.materialCount, cudaMemcpyKind::cudaMemcpyHostToDevice));

			gpuErrchk(cudaMemcpy(din.meshRendererBuffer + i, &c, sizeof(MeshRendererChunk), cudaMemcpyKind::cudaMemcpyHostToDevice));
		}

		gpuErrchk(cudaMalloc(&din.skinnedMeshRendererBuffer, sizeof(SkinnedMeshRendererChunk) * hin.skinnedMeshRendererBufferLen));
		for (int i = 0; i < hin.skinnedMeshRendererBufferLen; i++)
		{
			SkinnedMeshRendererChunk c = hin.skinnedMeshRendererBuffer[i], c2 = c;

			gpuErrchk(cudaMalloc(&c.materialArrayPtr, sizeof(int) * c.materialCount));
			gpuErrchk(cudaMemcpy(c.materialArrayPtr, c2.materialArrayPtr, sizeof(int) * c.materialCount, cudaMemcpyKind::cudaMemcpyHostToDevice));

			gpuErrchk(cudaMalloc(&c.boneArrayPtr, sizeof(Bone) * c.materialCount));
			gpuErrchk(cudaMemcpy(c.materialArrayPtr, c2.materialArrayPtr, sizeof(Bone) * c.materialCount, cudaMemcpyKind::cudaMemcpyHostToDevice));

			gpuErrchk(cudaMemcpy(din.skinnedMeshRendererBuffer + i, &c, sizeof(SkinnedMeshRendererChunk), cudaMemcpyKind::cudaMemcpyHostToDevice));
		}

		if (deviceIn != nullptr)
			gpuErrchk(cudaMemcpy(deviceIn, &din, sizeof(FrameMutableInput), cudaMemcpyKind::cudaMemcpyHostToDevice));

		return din;
	}

	__host__ void FreeDeviceImmutableData(FrameImmutableInput* deviceInputInHost)
	{
		for (int i = 0; i < deviceInputInHost->meshBufferLen; i++)
		{
			MeshChunk c;
			gpuErrchk(cudaMemcpy(&c, deviceInputInHost->meshBuffer + i, sizeof(MeshChunk), cudaMemcpyKind::cudaMemcpyDeviceToHost));

			gpuErrchk(cudaFree(c.positions));
			gpuErrchk(cudaFree(c.normals));
			gpuErrchk(cudaFree(c.uvs));
			gpuErrchk(cudaFree(c.indices));
			gpuErrchk(cudaFree(c.bindposeArrayPtr));
			gpuErrchk(cudaFree(c.submeshArrayPtr));
		}

		for (int i = 0; i < deviceInputInHost->skinnedMeshBufferLen; i++)
		{
			MeshChunk c;
			gpuErrchk(cudaMemcpy(&c, deviceInputInHost->skinnedMeshBuffer + i, sizeof(MeshChunk), cudaMemcpyKind::cudaMemcpyDeviceToHost));

			gpuErrchk(cudaFree(c.positions));
			gpuErrchk(cudaFree(c.normals));
			gpuErrchk(cudaFree(c.uvs));
			gpuErrchk(cudaFree(c.indices));
			gpuErrchk(cudaFree(c.bindposeArrayPtr));
			gpuErrchk(cudaFree(c.submeshArrayPtr));
		}

		for (int i = 0; i < deviceInputInHost->textureBufferLen; i++)
		{
			Texture2DChunk c;
			gpuErrchk(cudaMemcpy(&c, deviceInputInHost->textureBuffer + i, sizeof(Texture2DChunk), cudaMemcpyKind::cudaMemcpyDeviceToHost));

			struct cudaResourceDesc resDesc;
			gpuErrchk(cudaGetTextureObjectResourceDesc(&resDesc, reinterpret_cast<cudaTextureObject_t>(c.pixelPtr)));
			gpuErrchk(cudaFreeArray(resDesc.res.array.array));

			gpuErrchk(cudaDestroyTextureObject(reinterpret_cast<cudaTextureObject_t>(c.pixelPtr)));
		}

		gpuErrchk(cudaFree(deviceInputInHost->meshBuffer));
		gpuErrchk(cudaFree(deviceInputInHost->skinnedMeshBuffer));
		gpuErrchk(cudaFree(deviceInputInHost->textureBuffer));
	}

	__host__ void FreeDeviceMutableData(FrameMutableInput* deviceInputInHost)
	{
		for (int i = 0; i < deviceInputInHost->skinnedMeshRendererBufferLen; i++)
		{
			SkinnedMeshRendererChunk c;
			gpuErrchk(cudaMemcpy(&c, deviceInputInHost->skinnedMeshRendererBuffer + i, sizeof(SkinnedMeshRendererChunk), cudaMemcpyKind::cudaMemcpyDeviceToHost));

			gpuErrchk(cudaFree(c.materialArrayPtr));
			gpuErrchk(cudaFree(c.boneArrayPtr));
		}

		for (int i = 0; i < deviceInputInHost->meshRendererBufferLen; i++)
		{
			MeshRendererChunk c;
			gpuErrchk(cudaMemcpy(&c, deviceInputInHost->meshRendererBuffer + i, sizeof(MeshRendererChunk), cudaMemcpyKind::cudaMemcpyDeviceToHost));

			gpuErrchk(cudaFree(c.materialArrayPtr));
		}

		gpuErrchk(cudaFree(deviceInputInHost->cameraBuffer));
		gpuErrchk(cudaFree(deviceInputInHost->lightBuffer));
		gpuErrchk(cudaFree(deviceInputInHost->materialBuffer));
		
		gpuErrchk(cudaFree(deviceInputInHost->meshRendererBuffer));
		gpuErrchk(cudaFree(deviceInputInHost->skinnedMeshRendererBuffer));
		gpuErrchk(cudaFree(deviceInputInHost->skyboxMaterialBuffer));
	}

#pragma warning( push )
#pragma warning( disable : 4101 )

	__host__ void FreeHostMutableData(FrameMutableInput* hostIn)
	{
		for (int i = 0; i < hostIn->meshRendererBufferLen; i++)
		{
			MeshRendererChunk c = hostIn->meshRendererBuffer[i];
			SAFE_HOST_FREE(c.materialArrayPtr);
		}

		for (int i = 0; i < hostIn->skinnedMeshRendererBufferLen; i++)
		{
			SkinnedMeshRendererChunk c = hostIn->skinnedMeshRendererBuffer[i];
			SAFE_HOST_FREE(c.materialArrayPtr);
			SAFE_HOST_FREE(c.boneArrayPtr);
		}

		SAFE_HOST_FREE(hostIn->cameraBuffer);
		SAFE_HOST_FREE(hostIn->lightBuffer);
		SAFE_HOST_FREE(hostIn->materialBuffer);
		SAFE_HOST_FREE(hostIn->meshRendererBuffer);
		SAFE_HOST_FREE(hostIn->skinnedMeshRendererBuffer);
		SAFE_HOST_FREE(hostIn->skyboxMaterialBuffer);
	}
	__host__ void FreeHostImmutableData(FrameImmutableInput* hostIn)
	{
		for (int i = 0; i < hostIn->meshBufferLen; i++)
		{
			MeshChunk c = hostIn->meshBuffer[i];

			SAFE_HOST_FREE(c.positions);
			SAFE_HOST_FREE(c.normals);
			SAFE_HOST_FREE(c.uvs);
			SAFE_HOST_FREE(c.indices);
			SAFE_HOST_FREE(c.bindposeArrayPtr);
			SAFE_HOST_FREE(c.submeshArrayPtr);
		}

		for (int i = 0; i < hostIn->skinnedMeshBufferLen; i++)
		{
			MeshChunk c = hostIn->skinnedMeshBuffer[i];

			SAFE_HOST_FREE(c.positions);
			SAFE_HOST_FREE(c.normals);
			SAFE_HOST_FREE(c.uvs);
			SAFE_HOST_FREE(c.indices);
			SAFE_HOST_FREE(c.bindposeArrayPtr);
			SAFE_HOST_FREE(c.submeshArrayPtr);
		}

		for (int i = 0; i < hostIn->textureBufferLen; i++)
		{
			Texture2DChunk c = hostIn->textureBuffer[i];
			SAFE_HOST_FREE(c.pixelPtr);
		}

		SAFE_HOST_FREE(hostIn->meshBuffer);
		SAFE_HOST_FREE(hostIn->skinnedMeshBuffer);
		SAFE_HOST_FREE(hostIn->textureBuffer);
	}

#pragma warning( pop )

	__host__ size_t StoreMeshChunk(MeshChunk* m, FILE* fp)
	{
		size_t size = 0;
		size += fwrite(&m->vertexCount, sizeof(int), 1, fp);
		size += fwrite(&m->indexCount, sizeof(int), 1, fp);
		size += fwrite(&m->submeshCount, sizeof(int), 1, fp);
		size += fwrite(&m->bindposeCount, sizeof(int), 1, fp);
		int uvExistFlag = m->uvs != nullptr;
		size += fwrite(&uvExistFlag, sizeof(int), 1, fp);

		if (m->vertexCount > 0)
		{
			size += fwrite(m->positions, sizeof(Vector3f), m->vertexCount, fp);
			size += fwrite(m->normals, sizeof(Vector3f), m->vertexCount, fp);
			size += fwrite(m->tangents, sizeof(Vector4f), m->vertexCount, fp);

			if (m->uvs)
				size += fwrite(m->uvs, sizeof(Vector2f), m->vertexCount, fp);
		}
		if (m->indexCount > 0)
			size += fwrite(m->indices, sizeof(int), m->indexCount, fp);
		if (m->submeshCount > 0)
			size += fwrite(m->submeshArrayPtr, sizeof(UnitySubMeshDescriptor), m->submeshCount, fp);
		if (m->bindposeCount > 0)
			size += fwrite(m->bindposeArrayPtr, sizeof(Matrix4x4), m->bindposeCount, fp);

		size += fwrite(&m->aabbInMS, sizeof(Bounds), 1, fp);

		return size;
	}
	__host__ size_t StoreTexture2DChunk(Texture2DChunk* t, FILE* fp)
	{
		size_t size = 0;
		size += fwrite(&t->size, sizeof(Vector2i), 1, fp);
		size += fwrite(&t->filter, sizeof(eUnityFilterMode), 1, fp);
		size += fwrite(&t->anisotropic, sizeof(int), 1, fp);
		size += fwrite(t->pixelPtr, sizeof(ColorRGBA32), t->size.x * t->size.y, fp);
		size += fwrite(&t->hasAlpha, sizeof(bool), 1, fp);

		return size;
	}
	__host__ size_t StoreMeshRendererChunk(MeshRendererChunk* mr, FILE* fp)
	{
		size_t size = 0;
		size += fwrite(&mr->position, sizeof(Vector3f), 1, fp);
		size += fwrite(&mr->quaternion, sizeof(Quaternion), 1, fp);
		size += fwrite(&mr->scale, sizeof(Vector3f), 1, fp);
		size += fwrite(&mr->transformMatrix, sizeof(Matrix4x4), 1, fp);
		size += fwrite(&mr->transformInverseMatrix, sizeof(Matrix4x4), 1, fp);
		size += fwrite(&mr->meshRefIndex, sizeof(int), 1, fp);
		size += fwrite(&mr->boundingBox, sizeof(Bounds), 1, fp);
		size += fwrite(&mr->materialCount, sizeof(int), 1, fp);

		if (mr->materialCount > 0)
			size += fwrite(mr->materialArrayPtr, sizeof(int), mr->materialCount, fp);

		return size;
	}
	__host__ size_t StoreSkinnedMeshRendererChunk(SkinnedMeshRendererChunk* smr, FILE* fp)
	{
		size_t size = 0;
		size += fwrite(&smr->position, sizeof(Vector3f), 1, fp);
		size += fwrite(&smr->quaternion, sizeof(Quaternion), 1, fp);
		size += fwrite(&smr->scale, sizeof(Vector3f), 1, fp);
		size += fwrite(&smr->skinnedMeshRefIndex, sizeof(int), 1, fp);
		size += fwrite(&smr->boundingBox, sizeof(Bounds), 1, fp);
		size += fwrite(&smr->materialCount, sizeof(int), 1, fp);
		if (smr->materialCount > 0)
			size += fwrite(smr->materialArrayPtr, sizeof(int), smr->materialCount, fp);
		size += fwrite(&smr->boneCount, sizeof(int), 1, fp);
		if (smr->boneCount > 0)
			size += fwrite(smr->boneArrayPtr, sizeof(int), smr->boneCount, fp);

		return size;
	}
	__host__ size_t StoreMutableData(FrameMutableInput* hostIn, FILE* fp)
	{
		size_t size = 0;
		size += fwrite(&hostIn->meshRendererBufferLen, sizeof(int), 1, fp);
		for (int i = 0; i < hostIn->meshRendererBufferLen; i++)
			size += StoreMeshRendererChunk(hostIn->meshRendererBuffer + i, fp);

		size += fwrite(&hostIn->skinnedMeshRendererBufferLen, sizeof(int), 1, fp);
		for (int i = 0; i < hostIn->skinnedMeshRendererBufferLen; i++)
			size += StoreSkinnedMeshRendererChunk(hostIn->skinnedMeshRendererBuffer + i, fp);

		size += fwrite(&hostIn->lightBufferLen, sizeof(int), 1, fp);
		size += fwrite(hostIn->lightBuffer, sizeof(LightChunk), hostIn->lightBufferLen, fp);

		size += fwrite(&hostIn->cameraBufferLen, sizeof(int), 1, fp);
		size += fwrite(hostIn->cameraBuffer, sizeof(CameraChunk), hostIn->cameraBufferLen, fp);

		size += fwrite(&hostIn->skyboxMaterialBufferLen, sizeof(int), 1, fp);
		size += fwrite(hostIn->skyboxMaterialBuffer, sizeof(SkyboxChunk), hostIn->skyboxMaterialBufferLen, fp);

		size += fwrite(&hostIn->materialBufferLen, sizeof(int), 1, fp);
		size += fwrite(hostIn->materialBuffer, sizeof(MaterialChunk), hostIn->materialBufferLen, fp);

		size += fwrite(&hostIn->selectedCameraIndex, sizeof(int), 1, fp);

		return size;
	}
	__host__ size_t StoreImmutableData(FrameImmutableInput* hostIn, FILE* fp)
	{
		size_t size = 0;

		size += fwrite(&hostIn->meshBufferLen, sizeof(int), 1, fp);
		for (int i = 0; i < hostIn->meshBufferLen; i++)
			size += StoreMeshChunk(hostIn->meshBuffer + i, fp);

		size += fwrite(&hostIn->skinnedMeshBufferLen, sizeof(int), 1, fp);
		for (int i = 0; i < hostIn->skinnedMeshBufferLen; i++)
			size += StoreMeshChunk(hostIn->skinnedMeshBuffer + i, fp);

		size += fwrite(&hostIn->textureBufferLen, sizeof(int), 1, fp);
		for (int i = 0; i < hostIn->textureBufferLen; i++)
			size += StoreTexture2DChunk(hostIn->textureBuffer + i, fp);

		return size;
	}

	__host__ void LoadMeshChunk(MeshChunk* m, FILE* fp, void* (*allocator)(size_t cb))
	{
		int uvExistFlag;
		fread(&m->vertexCount, sizeof(int), 1, fp);
		fread(&m->indexCount, sizeof(int), 1, fp);
		fread(&m->submeshCount, sizeof(int), 1, fp);
		fread(&m->bindposeCount, sizeof(int), 1, fp);
		fread(&uvExistFlag, sizeof(int), 1, fp);

		m->positions = (Vector3f*)allocator(sizeof(Vector3f) * m->vertexCount);
		fread(m->positions, sizeof(Vector3f), m->vertexCount, fp);
		m->normals = (Vector3f*)allocator(sizeof(Vector3f) * m->vertexCount);
		fread(m->normals, sizeof(Vector3f), m->vertexCount, fp);
		m->tangents = (Vector4f*)allocator(sizeof(Vector4f) * m->vertexCount);
		fread(m->tangents, sizeof(Vector4f), m->vertexCount, fp);
		if (uvExistFlag)
		{
			m->uvs = (Vector2f*)allocator(sizeof(Vector2f) * m->vertexCount);
			fread(m->uvs, sizeof(Vector2f), m->vertexCount, fp);
		}
		else
			m->uvs = nullptr;
		m->indices = (int*)allocator(sizeof(int) * m->indexCount);
		fread(m->indices, sizeof(int), m->indexCount, fp);
		m->submeshArrayPtr = (UnitySubMeshDescriptor*)allocator(sizeof(UnitySubMeshDescriptor) * m->submeshCount);
		fread(m->submeshArrayPtr, sizeof(UnitySubMeshDescriptor), m->submeshCount, fp);
		for (int i = 0; i < m->submeshCount; i++)
			Log("%d::%d~%d", i, m->submeshArrayPtr[i].indexStart, m->submeshArrayPtr[i].indexCount);
		m->bindposeArrayPtr = (Matrix4x4*)allocator(sizeof(Matrix4x4) * m->bindposeCount);
		fread(m->bindposeArrayPtr, sizeof(Matrix4x4), m->bindposeCount, fp);

		fread(&m->aabbInMS, sizeof(Bounds), 1, fp);
	}
	__host__ void LoadTexture2DChunk(Texture2DChunk* t, FILE* fp, void* (*allocator)(size_t cb))
	{
		fread(&t->size, sizeof(Vector2i), 1, fp);
		fread(&t->filter, sizeof(eUnityFilterMode), 1, fp);
		fread(&t->anisotropic, sizeof(int), 1, fp);
		void* ptr = t->pixelPtr = malloc(sizeof(ColorRGBA32) * t->size.x * t->size.y);
		fread(ptr, sizeof(ColorRGBA32), t->size.x * t->size.y, fp);
		fread(&t->hasAlpha, sizeof(bool), 1, fp);
	}
	__host__ void LoadMeshRendererChunk(MeshRendererChunk* mr, FILE* fp, void* (*allocator)(size_t cb))
	{
		fread(&mr->position, sizeof(Vector3f), 1, fp);
		fread(&mr->quaternion, sizeof(Quaternion), 1, fp);
		fread(&mr->scale, sizeof(Vector3f), 1, fp);
		fread(&mr->transformMatrix, sizeof(Matrix4x4), 1, fp);
		fread(&mr->transformInverseMatrix, sizeof(Matrix4x4), 1, fp);
		fread(&mr->meshRefIndex, sizeof(int), 1, fp);
		fread(&mr->boundingBox, sizeof(Bounds), 1, fp);
		fread(&mr->materialCount, sizeof(int), 1, fp);
		mr->materialArrayPtr = (int*)allocator(sizeof(int) * mr->materialCount);
		fread(mr->materialArrayPtr, sizeof(int), mr->materialCount, fp);
	}
	__host__ void LoadSkinnedMeshRendererChunk(SkinnedMeshRendererChunk* smr, FILE* fp, void* (*allocator)(size_t cb))
	{
		fread(&smr->position, sizeof(Vector3f), 1, fp);
		fread(&smr->quaternion, sizeof(Quaternion), 1, fp);
		fread(&smr->scale, sizeof(Vector3f), 1, fp);
		fread(&smr->transformMatrix, sizeof(Matrix4x4), 1, fp);
		fread(&smr->transformInverseMatrix, sizeof(Matrix4x4), 1, fp);
		fread(&smr->skinnedMeshRefIndex, sizeof(int), 1, fp);
		fread(&smr->boundingBox, sizeof(Bounds), 1, fp);
		fread(&smr->materialCount, sizeof(int), 1, fp);
		smr->materialArrayPtr = (int*)allocator(sizeof(int) * smr->materialCount);
		fread(smr->materialArrayPtr, sizeof(int), smr->materialCount, fp);
		fread(&smr->boneCount, sizeof(int), 1, fp);
		smr->boneArrayPtr = (int*)allocator(sizeof(int) * smr->boneCount);
		fread(smr->boneArrayPtr, sizeof(int), smr->boneCount, fp);
	}
	__host__ void LoadMutableData(FrameMutableInput* hostIn, FILE* fp, void* (*allocator)(size_t cb))
	{
		fread(&hostIn->meshRendererBufferLen, sizeof(int), 1, fp);
		hostIn->meshRendererBuffer = (MeshRendererChunk*)allocator(sizeof(MeshRendererChunk) * hostIn->meshRendererBufferLen);
		for (int i = 0; i < hostIn->meshRendererBufferLen; i++)
			LoadMeshRendererChunk(hostIn->meshRendererBuffer + i, fp, allocator);

		fread(&hostIn->skinnedMeshRendererBufferLen, sizeof(int), 1, fp);
		hostIn->skinnedMeshRendererBuffer = (SkinnedMeshRendererChunk*)allocator(sizeof(SkinnedMeshRendererChunk) * hostIn->skinnedMeshRendererBufferLen);
		for (int i = 0; i < hostIn->skinnedMeshRendererBufferLen; i++)
			LoadSkinnedMeshRendererChunk(hostIn->skinnedMeshRendererBuffer + i, fp, allocator);

		fread(&hostIn->lightBufferLen, sizeof(int), 1, fp);
		hostIn->lightBuffer = (LightChunk*)allocator(sizeof(LightChunk) * hostIn->lightBufferLen);
		fread(hostIn->lightBuffer, sizeof(LightChunk), hostIn->lightBufferLen, fp);

		fread(&hostIn->cameraBufferLen, sizeof(int), 1, fp);
		hostIn->cameraBuffer = (CameraChunk*)allocator(sizeof(CameraChunk) * hostIn->cameraBufferLen);
		fread(hostIn->cameraBuffer, sizeof(CameraChunk), hostIn->cameraBufferLen, fp);

		fread(&hostIn->skyboxMaterialBufferLen, sizeof(int), 1, fp);
		hostIn->skyboxMaterialBuffer = (SkyboxChunk*)allocator(sizeof(SkyboxChunk) * hostIn->skyboxMaterialBufferLen);
		fread(hostIn->skyboxMaterialBuffer, sizeof(SkyboxChunk), hostIn->skyboxMaterialBufferLen, fp);

		fread(&hostIn->materialBufferLen, sizeof(int), 1, fp);
		hostIn->materialBuffer = (MaterialChunk*)allocator(sizeof(MaterialChunk) * hostIn->materialBufferLen);
		fread(hostIn->materialBuffer, sizeof(MaterialChunk), hostIn->materialBufferLen, fp);

		fread(&hostIn->selectedCameraIndex, sizeof(int), 1, fp);
	}
	__host__ void LoadImmutableData(FrameImmutableInput* hostIn, FILE* fp, void* (*allocator)(size_t cb))
	{
		fread(&hostIn->meshBufferLen, sizeof(int), 1, fp);
		hostIn->meshBuffer = (MeshChunk*)allocator(sizeof(MeshChunk) * hostIn->meshBufferLen);
		for (int i = 0; i < hostIn->meshBufferLen; i++)
			LoadMeshChunk(hostIn->meshBuffer + i, fp, allocator);

		fread(&hostIn->skinnedMeshBufferLen, sizeof(int), 1, fp);
		hostIn->skinnedMeshBuffer = (MeshChunk*)allocator(sizeof(MeshChunk) * hostIn->skinnedMeshBufferLen);
		for (int i = 0; i < hostIn->skinnedMeshBufferLen; i++)
			LoadMeshChunk(hostIn->skinnedMeshBuffer + i, fp, allocator);

		fread(&hostIn->textureBufferLen, sizeof(int), 1, fp);
		hostIn->textureBuffer = (Texture2DChunk*)allocator(sizeof(Texture2DChunk) * hostIn->textureBufferLen);
		for (int i = 0; i < hostIn->textureBufferLen; i++)
			LoadTexture2DChunk(hostIn->textureBuffer + i, fp, allocator);
	}

#pragma endregion IMPLEMENT_SUBROUTINES
}
