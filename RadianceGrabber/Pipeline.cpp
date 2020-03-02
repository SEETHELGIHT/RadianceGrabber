#include "Pipeline.h"

#include "Marshal.h"

namespace RadGrabber
{
	__host__ void AllocateDeviceMem(__host__ FrameRequest* hostReq, __device__ FrameInput** outDeviceInput)
	{
		ASSERT_IS_FALSE(outDeviceInput);

		FrameInput deviceInputBuffer = hostReq->input;
		ASSERT_IS_FALSE(cudaMalloc(outDeviceInput, sizeof(FrameInput)));

		ASSERT_IS_FALSE(cudaMalloc(&deviceInputBuffer.cameraBuffer, sizeof(CameraChunk) * hostReq->input.cameraBufferLen));
		ASSERT_IS_FALSE(cudaMalloc(&deviceInputBuffer.lightBuffer, sizeof(LightChunk) * hostReq->input.lightBufferLen));
		ASSERT_IS_FALSE(cudaMalloc(&deviceInputBuffer.materialBuffer, sizeof(MaterialChunk) * hostReq->input.materialBufferLen));
		ASSERT_IS_FALSE(cudaMalloc(&deviceInputBuffer.meshBuffer, sizeof(MeshChunk) * hostReq->input.meshBufferLen));
		ASSERT_IS_FALSE(cudaMalloc(&deviceInputBuffer.meshRendererBuffer, sizeof(MeshRendererChunk) * hostReq->input.meshRendererBufferLen));
		ASSERT_IS_FALSE(cudaMalloc(&deviceInputBuffer.skinnedMeshBuffer, sizeof(MeshChunk) * hostReq->input.skinnedMeshBufferLen));
		ASSERT_IS_FALSE(cudaMalloc(&deviceInputBuffer.skinnedMeshRendererBuffer, sizeof(SkinnedMeshRendererChunk) * hostReq->input.skinnedMeshRendererBufferLen));
		ASSERT_IS_FALSE(cudaMalloc(&deviceInputBuffer.skyboxMaterialBuffer, sizeof(SkyboxChunk) * hostReq->input.skyboxMaterialBufferLen));
		ASSERT_IS_FALSE(cudaMalloc(&deviceInputBuffer.textureBuffer, sizeof(Texture2DChunk) * hostReq->input.textureBufferLen));

		ASSERT_IS_FALSE(cudaMemcpy(*outDeviceInput, &hostReq->input, sizeof(FrameRequest), cudaMemcpyKind::cudaMemcpyHostToDevice));

		for (int i = 0; i < hostReq->input.meshBufferLen; i++)
		{
			MeshChunk c = hostReq->input.meshBuffer[i], c2 = c;

			ASSERT_IS_FALSE(cudaMalloc(&c.positions, sizeof(Vector3f) * c.vertexCount));
			ASSERT_IS_FALSE(cudaMemcpy(c.positions, c2.positions, sizeof(Vector3f) * c.vertexCount, cudaMemcpyKind::cudaMemcpyHostToDevice));
			ASSERT_IS_FALSE(cudaMalloc(&c.normals, sizeof(Vector3f) * c.vertexCount));
			ASSERT_IS_FALSE(cudaMemcpy(c.normals, c2.normals, sizeof(Vector3f) * c.vertexCount, cudaMemcpyKind::cudaMemcpyHostToDevice));
			ASSERT_IS_FALSE(cudaMalloc(&c.uvs, sizeof(Vector2f) * c.vertexCount));
			ASSERT_IS_FALSE(cudaMemcpy(c.uvs, c2.uvs, sizeof(Vector2f) * c.vertexCount, cudaMemcpyKind::cudaMemcpyHostToDevice));
			ASSERT_IS_FALSE(cudaMalloc(&c.indices, sizeof(int) * c.indexCount));
			ASSERT_IS_FALSE(cudaMemcpy(c.indices, c2.indices, sizeof(int) * c.indexCount, cudaMemcpyKind::cudaMemcpyHostToDevice));
			ASSERT_IS_FALSE(cudaMalloc(&c.submeshArrayPtr, sizeof(UnitySubMeshDescriptor) * c.submeshCount));
			ASSERT_IS_FALSE(cudaMemcpy(c.submeshArrayPtr, c2.submeshArrayPtr, sizeof(UnitySubMeshDescriptor) * c.submeshCount, cudaMemcpyKind::cudaMemcpyHostToDevice));

			if (c.bindposeCount > 0)
			{
				ASSERT_IS_FALSE(cudaMalloc(&c.bindposeArrayPtr, sizeof(Matrix4x4) * c.bindposeCount));
				ASSERT_IS_FALSE(cudaMemcpy(c.bindposeArrayPtr, c2.bindposeArrayPtr, sizeof(Matrix4x4) * c.bindposeCount, cudaMemcpyKind::cudaMemcpyHostToDevice));
			}
			else
				c.bindposeArrayPtr = nullptr;

			ASSERT_IS_FALSE(cudaMemcpy(deviceInputBuffer.meshBuffer + i, &c, sizeof(MeshChunk), cudaMemcpyKind::cudaMemcpyHostToDevice));
		}

		for (int i = 0; i < hostReq->input.skinnedMeshBufferLen; i++)
		{
			MeshChunk c = hostReq->input.skinnedMeshBuffer[i], c2 = c;

			ASSERT_IS_FALSE(cudaMalloc(&c.positions, sizeof(Vector3f) * c.vertexCount));
			ASSERT_IS_FALSE(cudaMemcpy(c.positions, c2.positions, sizeof(Vector3f) * c.vertexCount, cudaMemcpyKind::cudaMemcpyHostToDevice));
			ASSERT_IS_FALSE(cudaMalloc(&c.normals, sizeof(Vector3f) * c.vertexCount));
			ASSERT_IS_FALSE(cudaMemcpy(c.normals, c2.normals, sizeof(Vector3f) * c.vertexCount, cudaMemcpyKind::cudaMemcpyHostToDevice));
			ASSERT_IS_FALSE(cudaMalloc(&c.uvs, sizeof(Vector2f) * c.vertexCount));
			ASSERT_IS_FALSE(cudaMemcpy(c.uvs, c2.uvs, sizeof(Vector2f) * c.vertexCount, cudaMemcpyKind::cudaMemcpyHostToDevice));
			ASSERT_IS_FALSE(cudaMalloc(&c.indices, sizeof(int) * c.indexCount));
			ASSERT_IS_FALSE(cudaMemcpy(c.indices, c2.indices, sizeof(int) * c.indexCount, cudaMemcpyKind::cudaMemcpyHostToDevice));
			ASSERT_IS_FALSE(cudaMalloc(&c.submeshArrayPtr, sizeof(UnitySubMeshDescriptor) * c.submeshCount));
			ASSERT_IS_FALSE(cudaMemcpy(c.submeshArrayPtr, c2.submeshArrayPtr, sizeof(UnitySubMeshDescriptor) * c.submeshCount, cudaMemcpyKind::cudaMemcpyHostToDevice));

			if (c.bindposeCount > 0)
			{
				ASSERT_IS_FALSE(cudaMalloc(&c.bindposeArrayPtr, sizeof(Matrix4x4) * c.bindposeCount));
				ASSERT_IS_FALSE(cudaMemcpy(c.bindposeArrayPtr, c2.bindposeArrayPtr, sizeof(Matrix4x4) * c.bindposeCount, cudaMemcpyKind::cudaMemcpyHostToDevice));
			}
			else
				c.bindposeArrayPtr = nullptr;

			ASSERT_IS_FALSE(cudaMemcpy(deviceInputBuffer.skinnedMeshBuffer + i, &c, sizeof(MeshChunk), cudaMemcpyKind::cudaMemcpyHostToDevice));
		}

		for (int i = 0; i < hostReq->input.meshRendererBufferLen; i++)
		{
			MeshRendererChunk c = hostReq->input.meshRendererBuffer[i], c2 = c;

			ASSERT_IS_FALSE(cudaMalloc(&c.materialArrayPtr, sizeof(int) * c.materialCount));
			ASSERT_IS_FALSE(cudaMemcpy(c.materialArrayPtr, c2.materialArrayPtr, sizeof(int) * c.materialCount, cudaMemcpyKind::cudaMemcpyHostToDevice));

			ASSERT_IS_FALSE(cudaMemcpy(deviceInputBuffer.meshRendererBuffer + i, &c, sizeof(MeshRendererChunk), cudaMemcpyKind::cudaMemcpyHostToDevice));
		}

		for (int i = 0; i < hostReq->input.skinnedMeshRendererBufferLen; i++)
		{
			SkinnedMeshRendererChunk c = hostReq->input.skinnedMeshRendererBuffer[i], c2 = c;

			ASSERT_IS_FALSE(cudaMalloc(&c.materialArrayPtr, sizeof(int) * c.materialCount));
			ASSERT_IS_FALSE(cudaMemcpy(c.materialArrayPtr, c2.materialArrayPtr, sizeof(int) * c.materialCount, cudaMemcpyKind::cudaMemcpyHostToDevice));

			ASSERT_IS_FALSE(cudaMalloc(&c.boneArrayPtr, sizeof(Bone) * c.materialCount));
			ASSERT_IS_FALSE(cudaMemcpy(c.materialArrayPtr, c2.materialArrayPtr, sizeof(Bone) * c.materialCount, cudaMemcpyKind::cudaMemcpyHostToDevice));

			ASSERT_IS_FALSE(cudaMemcpy(deviceInputBuffer.skinnedMeshRendererBuffer + i, &c, sizeof(SkinnedMeshRendererChunk), cudaMemcpyKind::cudaMemcpyHostToDevice));
		}

		for (int i = 0; i < hostReq->input.textureBufferLen; i++)
		{
			Texture2DChunk c = hostReq->input.textureBuffer[i], c2 = c;

			cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
			cudaArray* pixelArray = nullptr;
			ASSERT_IS_FALSE(cudaMallocArray(&pixelArray, &channelDesc, c.size.x, c.size.y));
			ASSERT_IS_FALSE(cudaMemcpyToArray(pixelArray, 0, 0, c2.pixelPtr, sizeof(Vector4f) * c.size.x * c.size.y, cudaMemcpyKind::cudaMemcpyHostToDevice));

			/*
				TODO
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

			ASSERT_IS_FALSE(sizeof(cudaTextureObject_t) == sizeof(void*));
			cudaCreateTextureObject((cudaTextureObject_t*)&c.pixelPtr, &resDesc, &texDesc, nullptr);

			ASSERT_IS_FALSE(cudaMemcpy(deviceInputBuffer.textureBuffer + i, &c, sizeof(Texture2DChunk), cudaMemcpyKind::cudaMemcpyHostToDevice));
		}
	}

#pragma warning( push )
#pragma warning( disable : 4101 )
	__host__ void FreeDeviceMem(FrameInput* deviceInput)
	{
		FrameInput input;
		ASSERT_IS_FALSE(cudaMemcpy(&input, deviceInput, sizeof(FrameInput), cudaMemcpyKind::cudaMemcpyDeviceToHost));

		for (int i = 0; i < input.meshBufferLen; i++)
		{
			MeshChunk c;
			ASSERT_IS_FALSE(cudaMemcpy(&c, input.meshBuffer + i, sizeof(MeshChunk), cudaMemcpyKind::cudaMemcpyDeviceToHost));

			ASSERT_IS_FALSE(cudaFree(c.positions));
			ASSERT_IS_FALSE(cudaFree(c.normals));
			ASSERT_IS_FALSE(cudaFree(c.uvs));
			ASSERT_IS_FALSE(cudaFree(c.indices));
			ASSERT_IS_FALSE(cudaFree(c.bindposeArrayPtr));
			ASSERT_IS_FALSE(cudaFree(c.submeshArrayPtr));
		}

		for (int i = 0; i < input.skinnedMeshBufferLen; i++)
		{
			MeshChunk c;
			ASSERT_IS_FALSE(cudaMemcpy(&c, input.skinnedMeshBuffer + i, sizeof(MeshChunk), cudaMemcpyKind::cudaMemcpyDeviceToHost));

			ASSERT_IS_FALSE(cudaFree(c.positions));
			ASSERT_IS_FALSE(cudaFree(c.normals));
			ASSERT_IS_FALSE(cudaFree(c.uvs));
			ASSERT_IS_FALSE(cudaFree(c.indices));
			ASSERT_IS_FALSE(cudaFree(c.bindposeArrayPtr));
			ASSERT_IS_FALSE(cudaFree(c.submeshArrayPtr));
		}

		for (int i = 0; i < input.meshRendererBufferLen; i++)
		{
			MeshRendererChunk c;
			ASSERT_IS_FALSE(cudaMemcpy(&c, input.meshRendererBuffer + i, sizeof(MeshRendererChunk), cudaMemcpyKind::cudaMemcpyDeviceToHost));

			ASSERT_IS_FALSE(cudaFree(c.materialArrayPtr));
		}

		for (int i = 0; i < input.skinnedMeshRendererBufferLen; i++)
		{
			SkinnedMeshRendererChunk c;
			ASSERT_IS_FALSE(cudaMemcpy(&c, input.skinnedMeshRendererBuffer + i, sizeof(SkinnedMeshRendererChunk), cudaMemcpyKind::cudaMemcpyDeviceToHost));

			ASSERT_IS_FALSE(cudaFree(c.materialArrayPtr));
			ASSERT_IS_FALSE(cudaFree(c.boneArrayPtr));
		}

		for (int i = 0; i < input.textureBufferLen; i++)
		{
			Texture2DChunk c;
			ASSERT_IS_FALSE(cudaMemcpy(&c, input.textureBuffer + i, sizeof(Texture2DChunk), cudaMemcpyKind::cudaMemcpyDeviceToHost));

			struct cudaResourceDesc resDesc;
			ASSERT_IS_FALSE(cudaGetTextureObjectResourceDesc(&resDesc, reinterpret_cast<cudaTextureObject_t>(c.pixelPtr)));
			ASSERT_IS_FALSE(cudaFreeArray(resDesc.res.array.array));

			ASSERT_IS_FALSE(cudaDestroyTextureObject(reinterpret_cast<cudaTextureObject_t>(c.pixelPtr)));
		}

		ASSERT_IS_FALSE(cudaFree(input.cameraBuffer));
		ASSERT_IS_FALSE(cudaFree(input.lightBuffer));
		ASSERT_IS_FALSE(cudaFree(input.materialBuffer));
		ASSERT_IS_FALSE(cudaFree(input.meshBuffer));
		ASSERT_IS_FALSE(cudaFree(input.meshRendererBuffer));
		ASSERT_IS_FALSE(cudaFree(input.skinnedMeshBuffer));
		ASSERT_IS_FALSE(cudaFree(input.skinnedMeshRendererBuffer));
		ASSERT_IS_FALSE(cudaFree(input.skyboxMaterialBuffer));
		ASSERT_IS_FALSE(cudaFree(input.textureBuffer));

		ASSERT_IS_FALSE(cudaFree(deviceInput));
	}
#pragma warning( pop )
}
