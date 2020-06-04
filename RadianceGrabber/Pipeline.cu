#include "Pipeline.h"

#include "Marshal.cuh"

/*
	TODO:: using custom memory allocator 
	ASSERT_IS_FALSE\(cudaMalloc\(\&*(.+)\,\ 
	$1 = MAllocDevice(
*/
namespace RadGrabber
{
	__global__ void SetDeviceMem(FrameInput* din, FrameInputInternal in)
	{
		if (threadIdx.x == 0)
			din = new (din)FrameInput(in);
	}

	__host__ void AllocateDeviceMem(FrameRequest* hostReq, FrameInput** outDeviceInput)
	{
		ASSERT(outDeviceInput);

		FrameInput hin = hostReq->input;
		FrameInputInternal dib = hostReq->input.in;
		gpuErrchk(cudaMalloc(outDeviceInput, sizeof(FrameInput)));

		gpuErrchk(cudaMalloc(&dib.meshBuffer, sizeof(MeshChunk) * hin.in.meshBufferLen));
		gpuErrchk(cudaMalloc(&dib.meshRendererBuffer, sizeof(MeshRendererChunk) * hin.in.meshRendererBufferLen));
		gpuErrchk(cudaMalloc(&dib.skinnedMeshBuffer, sizeof(MeshChunk) * hin.in.skinnedMeshBufferLen));
		gpuErrchk(cudaMalloc(&dib.skinnedMeshRendererBuffer, sizeof(SkinnedMeshRendererChunk) * hin.in.skinnedMeshRendererBufferLen));
		gpuErrchk(cudaMalloc(&dib.textureBuffer, sizeof(Texture2DChunk) * hin.in.textureBufferLen));

		gpuErrchk(cudaMalloc(&dib.cameraBuffer, sizeof(CameraChunk) * hin.in.cameraBufferLen));
		gpuErrchk(cudaMemcpy(dib.cameraBuffer, hin.in.cameraBuffer, sizeof(CameraChunk) * hin.in.cameraBufferLen, cudaMemcpyKind::cudaMemcpyHostToDevice));
		gpuErrchk(cudaMalloc(&dib.lightBuffer, sizeof(LightChunk) * hin.in.lightBufferLen));
		gpuErrchk(cudaMemcpy(dib.lightBuffer, hin.in.lightBuffer, sizeof(LightChunk) * hin.in.lightBufferLen, cudaMemcpyKind::cudaMemcpyHostToDevice));
		gpuErrchk(cudaMalloc(&dib.materialBuffer, sizeof(MaterialChunk) * hin.in.materialBufferLen));
		gpuErrchk(cudaMemcpy(dib.materialBuffer, hin.in.materialBuffer, sizeof(MaterialChunk) * hin.in.materialBufferLen, cudaMemcpyKind::cudaMemcpyHostToDevice));
		gpuErrchk(cudaMalloc(&dib.skyboxMaterialBuffer, sizeof(SkyboxChunk) * hin.in.skyboxMaterialBufferLen));
		gpuErrchk(cudaMemcpy(dib.skyboxMaterialBuffer, hin.in.skyboxMaterialBuffer, sizeof(SkyboxChunk) * hin.in.skyboxMaterialBufferLen, cudaMemcpyKind::cudaMemcpyHostToDevice));

		SetDeviceMem<<<1, 1>>>(*outDeviceInput, dib);

		for (int i = 0; i < hin.in.meshBufferLen; i++)
		{
			MeshChunk c = hin.in.meshBuffer[i], c2 = c;

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

			gpuErrchk(cudaMemcpy(dib.meshBuffer + i, &c, sizeof(MeshChunk), cudaMemcpyKind::cudaMemcpyHostToDevice));
		}

		for (int i = 0; i < hin.in.skinnedMeshBufferLen; i++)
		{
			MeshChunk c = hin.in.skinnedMeshBuffer[i], c2 = c;

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

			gpuErrchk(cudaMemcpy(dib.skinnedMeshBuffer + i, &c, sizeof(MeshChunk), cudaMemcpyKind::cudaMemcpyHostToDevice));
		}

		for (int i = 0; i < hin.in.meshRendererBufferLen; i++)
		{
			MeshRendererChunk c = hin.in.meshRendererBuffer[i], c2 = c;

			gpuErrchk(cudaMalloc(&c.materialArrayPtr, sizeof(int) * c.materialCount));
			gpuErrchk(cudaMemcpy(c.materialArrayPtr, c2.materialArrayPtr, sizeof(int) * c.materialCount, cudaMemcpyKind::cudaMemcpyHostToDevice));

			gpuErrchk(cudaMemcpy(dib.meshRendererBuffer + i, &c, sizeof(MeshRendererChunk), cudaMemcpyKind::cudaMemcpyHostToDevice));
		}

		for (int i = 0; i < hin.in.skinnedMeshRendererBufferLen; i++)
		{
			SkinnedMeshRendererChunk c = hin.in.skinnedMeshRendererBuffer[i], c2 = c;

			gpuErrchk(cudaMalloc(&c.materialArrayPtr, sizeof(int) * c.materialCount));
			gpuErrchk(cudaMemcpy(c.materialArrayPtr, c2.materialArrayPtr, sizeof(int) * c.materialCount, cudaMemcpyKind::cudaMemcpyHostToDevice));

			gpuErrchk(cudaMalloc(&c.boneArrayPtr, sizeof(Bone) * c.materialCount));
			gpuErrchk(cudaMemcpy(c.materialArrayPtr, c2.materialArrayPtr, sizeof(Bone) * c.materialCount, cudaMemcpyKind::cudaMemcpyHostToDevice));

			gpuErrchk(cudaMemcpy(dib.skinnedMeshRendererBuffer + i, &c, sizeof(SkinnedMeshRendererChunk), cudaMemcpyKind::cudaMemcpyHostToDevice));
		}

		for (int i = 0; i < hin.in.textureBufferLen; i++)
		{
			Texture2DChunk c = hin.in.textureBuffer[i], c2 = c;

			cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(8, 8, 8, 8, cudaChannelFormatKind::cudaChannelFormatKindUnsigned );
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

			gpuErrchk(cudaMemcpy(dib.textureBuffer + i, &c, sizeof(Texture2DChunk), cudaMemcpyKind::cudaMemcpyHostToDevice));
		}
	}

#pragma warning( push )
#pragma warning( disable : 4101 )
	__host__ void FreeDeviceMem(FrameInput* deviceInput)
	{
		FrameInput in;
		gpuErrchk(cudaMemcpy(&in, deviceInput, sizeof(FrameInput), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		for (int i = 0; i < in.in.meshBufferLen; i++)
		{
			MeshChunk c;
			gpuErrchk(cudaMemcpy(&c, in.in.meshBuffer + i, sizeof(MeshChunk), cudaMemcpyKind::cudaMemcpyDeviceToHost));

			gpuErrchk(cudaFree(c.positions));
			gpuErrchk(cudaFree(c.normals));
			gpuErrchk(cudaFree(c.uvs));
			gpuErrchk(cudaFree(c.indices));
			gpuErrchk(cudaFree(c.bindposeArrayPtr));
			gpuErrchk(cudaFree(c.submeshArrayPtr));
		}

		for (int i = 0; i < in.in.skinnedMeshBufferLen; i++)
		{
			MeshChunk c;
			gpuErrchk(cudaMemcpy(&c, in.in.skinnedMeshBuffer + i, sizeof(MeshChunk), cudaMemcpyKind::cudaMemcpyDeviceToHost));

			gpuErrchk(cudaFree(c.positions));
			gpuErrchk(cudaFree(c.normals));
			gpuErrchk(cudaFree(c.uvs));
			gpuErrchk(cudaFree(c.indices));
			gpuErrchk(cudaFree(c.bindposeArrayPtr));
			gpuErrchk(cudaFree(c.submeshArrayPtr));
		}

		for (int i = 0; i < in.in.meshRendererBufferLen; i++)
		{
			MeshRendererChunk c;
			gpuErrchk(cudaMemcpy(&c, in.in.meshRendererBuffer + i, sizeof(MeshRendererChunk), cudaMemcpyKind::cudaMemcpyDeviceToHost));

			gpuErrchk(cudaFree(c.materialArrayPtr));
		}

		for (int i = 0; i < in.in.skinnedMeshRendererBufferLen; i++)
		{
			SkinnedMeshRendererChunk c;
			gpuErrchk(cudaMemcpy(&c, in.in.skinnedMeshRendererBuffer + i, sizeof(SkinnedMeshRendererChunk), cudaMemcpyKind::cudaMemcpyDeviceToHost));

			gpuErrchk(cudaFree(c.materialArrayPtr));
			gpuErrchk(cudaFree(c.boneArrayPtr));
		}

		for (int i = 0; i < in.in.textureBufferLen; i++)
		{
			Texture2DChunk c;
			gpuErrchk(cudaMemcpy(&c, in.in.textureBuffer + i, sizeof(Texture2DChunk), cudaMemcpyKind::cudaMemcpyDeviceToHost));

			struct cudaResourceDesc resDesc;
			gpuErrchk(cudaGetTextureObjectResourceDesc(&resDesc, reinterpret_cast<cudaTextureObject_t>(c.pixelPtr)));
			gpuErrchk(cudaFreeArray(resDesc.res.array.array));

			gpuErrchk(cudaDestroyTextureObject(reinterpret_cast<cudaTextureObject_t>(c.pixelPtr)));
		}

		gpuErrchk(cudaFree(in.in.cameraBuffer));
		gpuErrchk(cudaFree(in.in.lightBuffer));
		gpuErrchk(cudaFree(in.in.materialBuffer));
		gpuErrchk(cudaFree(in.in.meshBuffer));
		gpuErrchk(cudaFree(in.in.meshRendererBuffer));
		gpuErrchk(cudaFree(in.in.skinnedMeshBuffer));
		gpuErrchk(cudaFree(in.in.skinnedMeshRendererBuffer));
		gpuErrchk(cudaFree(in.in.skyboxMaterialBuffer));
		gpuErrchk(cudaFree(in.in.textureBuffer));

		gpuErrchk(cudaFree(deviceInput));
	}

	__host__ void FreeHostMem(FrameRequest* hreq)
	{
		for (int i = 0; i < hreq->input.in.meshBufferLen; i++)
		{
			MeshChunk c = hreq->input.in.meshBuffer[i];

			SAFE_HOST_FREE(c.positions);
			SAFE_HOST_FREE(c.normals);
			SAFE_HOST_FREE(c.uvs);
			SAFE_HOST_FREE(c.indices);
			SAFE_HOST_FREE(c.bindposeArrayPtr);
			SAFE_HOST_FREE(c.submeshArrayPtr);
		}

		for (int i = 0; i < hreq->input.in.skinnedMeshBufferLen; i++)
		{
			MeshChunk c = hreq->input.in.skinnedMeshBuffer[i];

			SAFE_HOST_FREE(c.positions);
			SAFE_HOST_FREE(c.normals);
			SAFE_HOST_FREE(c.uvs);
			SAFE_HOST_FREE(c.indices);
			SAFE_HOST_FREE(c.bindposeArrayPtr);
			SAFE_HOST_FREE(c.submeshArrayPtr);
		}

		for (int i = 0; i < hreq->input.in.meshRendererBufferLen; i++)
		{
			MeshRendererChunk c = hreq->input.in.meshRendererBuffer[i];
			SAFE_HOST_FREE(c.materialArrayPtr);
		}

		for (int i = 0; i < hreq->input.in.skinnedMeshRendererBufferLen; i++)
		{
			SkinnedMeshRendererChunk c = hreq->input.in.skinnedMeshRendererBuffer[i];
			SAFE_HOST_FREE(c.materialArrayPtr);
			SAFE_HOST_FREE(c.boneArrayPtr);
		}

		for (int i = 0; i < hreq->input.in.textureBufferLen; i++)
		{
			Texture2DChunk c = hreq->input.in.textureBuffer[i];
			SAFE_HOST_FREE(c.pixelPtr);
		}

		SAFE_HOST_FREE(hreq->input.in.cameraBuffer);
		SAFE_HOST_FREE(hreq->input.in.lightBuffer);
		SAFE_HOST_FREE(hreq->input.in.materialBuffer);
		SAFE_HOST_FREE(hreq->input.in.meshBuffer);
		SAFE_HOST_FREE(hreq->input.in.meshRendererBuffer);
		SAFE_HOST_FREE(hreq->input.in.skinnedMeshBuffer);
		SAFE_HOST_FREE(hreq->input.in.skinnedMeshRendererBuffer);
		SAFE_HOST_FREE(hreq->input.in.skyboxMaterialBuffer);
		SAFE_HOST_FREE(hreq->input.in.textureBuffer);

		SAFE_HOST_FREE(hreq);
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

	__host__ size_t StoreFrameRequest(FrameRequest* hostReq, FILE* fp)
	{
		size_t size = 0;
		{
			size += fwrite(&hostReq->opt, sizeof(RequestOption), 1, fp);
		}

		{
			FrameInputInternal& in = hostReq->input.in;

			size += fwrite(&in.meshBufferLen, sizeof(int), 1, fp);
			for (int i = 0; i < in.meshBufferLen; i++)
				size += StoreMeshChunk(in.meshBuffer + i, fp);

			size += fwrite(&in.skinnedMeshBufferLen, sizeof(int), 1, fp);
			for (int i = 0; i < in.skinnedMeshBufferLen; i++)
				size += StoreMeshChunk(in.skinnedMeshBuffer + i, fp);

			size += fwrite(&in.textureBufferLen, sizeof(int), 1, fp);
			for (int i = 0; i < in.textureBufferLen; i++)
				size += StoreTexture2DChunk(in.textureBuffer + i, fp);

			size += fwrite(&in.meshRendererBufferLen, sizeof(int), 1, fp);
			for (int i = 0; i < in.meshRendererBufferLen; i++)
				size += StoreMeshRendererChunk(in.meshRendererBuffer + i, fp);

			size += fwrite(&in.skinnedMeshRendererBufferLen, sizeof(int), 1, fp);
			for (int i = 0; i < in.skinnedMeshRendererBufferLen; i++)
				size += StoreSkinnedMeshRendererChunk(in.skinnedMeshRendererBuffer + i, fp);

			size += fwrite(&in.lightBufferLen, sizeof(int), 1, fp);
			size += fwrite(in.lightBuffer, sizeof(LightChunk), in.lightBufferLen, fp);

			size += fwrite(&in.cameraBufferLen, sizeof(int), 1, fp);
			size += fwrite(in.cameraBuffer, sizeof(CameraChunk), in.cameraBufferLen, fp);

			size += fwrite(&in.skyboxMaterialBufferLen, sizeof(int), 1, fp);
			size += fwrite(in.skyboxMaterialBuffer, sizeof(SkyboxChunk), in.skyboxMaterialBufferLen, fp);
			
			size += fwrite(&in.materialBufferLen, sizeof(int), 1, fp);
			size += fwrite(in.materialBuffer, sizeof(MaterialChunk), in.materialBufferLen, fp);

			size += fwrite(&in.selectedCameraIndex, sizeof(int), 1, fp);
		}

		{
			size += fwrite(&hostReq->output, sizeof(FrameOutput), 1, fp);
		}

		fflush(fp);
		FlushLog();

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

	__host__ void LoadFrameRequest(FILE* fp, FrameRequest** reqBuffer, void* (*allocator)(size_t cb))
	{
		{
			fread(&(*reqBuffer)->opt, sizeof(RequestOption), 1, fp);
		}
		
		{
			FrameInput& in = (*reqBuffer)->input;

			fread(&in.in.meshBufferLen, sizeof(int), 1, fp);
			in.in.meshBuffer = (MeshChunk*)allocator(sizeof(MeshChunk) * in.in.meshBufferLen);
			for (int i = 0; i < in.in.meshBufferLen; i++)
				LoadMeshChunk(in.in.meshBuffer + i, fp, allocator);

			fread(&in.in.skinnedMeshBufferLen, sizeof(int), 1, fp);
			in.in.skinnedMeshBuffer = (MeshChunk*)allocator(sizeof(MeshChunk) * in.in.skinnedMeshBufferLen);
			for (int i = 0; i < in.in.skinnedMeshBufferLen; i++)
				LoadMeshChunk(in.in.skinnedMeshBuffer + i, fp, allocator);

			fread(&in.in.textureBufferLen, sizeof(int), 1, fp);
			in.in.textureBuffer = (Texture2DChunk*)allocator(sizeof(Texture2DChunk) * in.in.textureBufferLen);
			for (int i = 0; i < in.in.textureBufferLen; i++)
				LoadTexture2DChunk(in.in.textureBuffer + i, fp, allocator);

			fread(&in.in.meshRendererBufferLen, sizeof(int), 1, fp);
			in.in.meshRendererBuffer = (MeshRendererChunk*)allocator(sizeof(MeshRendererChunk) * in.in.meshRendererBufferLen);
			for (int i = 0; i < in.in.meshRendererBufferLen; i++)
				LoadMeshRendererChunk(in.in.meshRendererBuffer + i, fp, allocator);

			fread(&in.in.skinnedMeshRendererBufferLen, sizeof(int), 1, fp);
			in.in.skinnedMeshRendererBuffer = (SkinnedMeshRendererChunk*)allocator(sizeof(SkinnedMeshRendererChunk) * in.in.skinnedMeshRendererBufferLen);
			for (int i = 0; i < in.in.skinnedMeshRendererBufferLen; i++)
				LoadSkinnedMeshRendererChunk(in.in.skinnedMeshRendererBuffer + i, fp, allocator);

			fread(&in.in.lightBufferLen, sizeof(int), 1, fp);
			in.in.lightBuffer = (LightChunk*)allocator(sizeof(LightChunk) * in.in.lightBufferLen);
			fread(in.in.lightBuffer, sizeof(LightChunk), in.in.lightBufferLen, fp);

			fread(&in.in.cameraBufferLen, sizeof(int), 1, fp);
			in.in.cameraBuffer = (CameraChunk*)allocator(sizeof(CameraChunk) * in.in.cameraBufferLen);
			fread(in.in.cameraBuffer, sizeof(CameraChunk), in.in.cameraBufferLen, fp);

			fread(&in.in.skyboxMaterialBufferLen, sizeof(int), 1, fp);
			in.in.skyboxMaterialBuffer = (SkyboxChunk*)allocator(sizeof(SkyboxChunk) * in.in.skyboxMaterialBufferLen);
			fread(in.in.skyboxMaterialBuffer, sizeof(SkyboxChunk), in.in.skyboxMaterialBufferLen, fp);

			fread(&in.in.materialBufferLen, sizeof(int), 1, fp);
			in.in.materialBuffer = (MaterialChunk*)allocator(sizeof(MaterialChunk) * in.in.materialBufferLen);
			fread(in.in.materialBuffer, sizeof(MaterialChunk), in.in.materialBufferLen, fp);

			fread(&in.in.selectedCameraIndex, sizeof(int), 1, fp);
		}

		{
			fread(&(*reqBuffer)->output, sizeof(FrameOutput), 1, fp);
		}
	}
}
