#include "Mesh.h"
#include "Unity/RenderAPI.h"

#include <cuda_runtime.h>

namespace RadGrabber
{
	extern RenderAPI* s_CurrentAPI;

	__host__ Mesh::Mesh() : mPositions(nullptr), mNormals(nullptr), mUV1(nullptr)
	{
	}

	__host__ Mesh::~Mesh()
	{
		SAFE_DEVICE_DELETE(mPositions);
		SAFE_DEVICE_DELETE(mNormals);
		SAFE_DEVICE_DELETE(mUV1);
	}

	__host__ Mesh::Mesh(const Mesh & m)
	{
		ASSERT(CopyFromDevice(m))  ;
	}

	__host__ Mesh::Mesh(const Mesh && m)
	{
		mVertexCount = m.mVertexCount;
		mIndexCount = m.mIndexCount;

		mPositions = m.mPositions;
		mNormals = m.mNormals;
		mUV1 = m.mUV1;
	}

	__host__ Mesh::Mesh(const MeshChunk* mc)
	{
		ASSERT(BuildFromChunk(mc));
	}

	__host__ __device__ int GetSizeFromFormat(eUnityVertexAttributeFormat fmt)
	{
		switch (fmt)
		{
		case RadGrabber::eUnityVertexAttributeFormat::UNorm8:
		case RadGrabber::eUnityVertexAttributeFormat::SNorm8:
		case RadGrabber::eUnityVertexAttributeFormat::UInt8:
		case RadGrabber::eUnityVertexAttributeFormat::SInt8:
			return 1;			
		case RadGrabber::eUnityVertexAttributeFormat::UNorm16:
		case RadGrabber::eUnityVertexAttributeFormat::SNorm16:
		case RadGrabber::eUnityVertexAttributeFormat::UInt16:
		case RadGrabber::eUnityVertexAttributeFormat::SInt16:
		case RadGrabber::eUnityVertexAttributeFormat::Float16:
			return 2;
		case RadGrabber::eUnityVertexAttributeFormat::Float32:
		case RadGrabber::eUnityVertexAttributeFormat::UInt32:
		case RadGrabber::eUnityVertexAttributeFormat::SInt32:
			return 4;
		}

		return -1;
	}

	__host__ bool Mesh::BuildFromChunk(const MeshChunk* p)
	{
		ASSERT(cudaMallocManaged(&mPositions, sizeof(Vector3f) * mVertexCount));
		ASSERT(cudaMallocManaged(&mNormals, sizeof(Vector3f) * mVertexCount));
		ASSERT(cudaMallocManaged(&mUV1, sizeof(Vector2f) * mVertexCount));
		ASSERT(cudaMallocManaged(&mIndices, sizeof(int) * mIndexCount));

		int attrSize		= 0,	*attrOffsets	= (int*)alloca(sizeof(int) * p->vertexBufferCount), stream,
			positionOffset	= -1,	normalOffset	= -1,	uvOffset	= -1,
			positionSize	= -1,	normalSize		= -1,	uvSize		= -1,
			positionDim		= -1,	normalDim		= -1,	uvDim		= -1,
			positionStream	= -1,	normalStream	= -1,	uvStream	= -1;	

		memset(attrOffsets, 0, sizeof(int) * p->vertexBufferCount);

		for (int i = 0, attrOffset = 0, size = 0, dim = 0; i < p->vertexAttributeCount; i++)
		{
			attrSize += size = GetSizeFromFormat(p->vertexAttributePtr[i].format);
			dim = p->vertexAttributePtr[i].dimension;
			stream = p->vertexAttributePtr[i].stream;

			switch (p->vertexAttributePtr[i].attribute)
			{
			case eUnityVertexAttribute::Position:
				positionOffset = attrOffset;
				positionSize = size;
				positionDim = dim;
				positionStream = stream;

				ASSERT(positionSize >= 4 && positionDim >= 3);
				break;
			case eUnityVertexAttribute::Normal:
				normalOffset = attrOffset;
				normalSize = size;
				normalDim = dim;
				normalStream = stream;

				ASSERT(normalSize >= 4 && normalDim >= 2);
				break;
			case eUnityVertexAttribute::TexCoord0:
				uvOffset = attrOffset;
				uvSize = size;
				uvDim = dim;
				uvStream = stream;

				ASSERT(uvSize == 4 && uvDim == 2);
				break;
			}

			attrOffsets[stream] += p->vertexAttributePtr[i].dimension * size * dim;
		}

		int size;
		byte* vb = nullptr;
		
		vb = (byte*)s_CurrentAPI->BeginReadVertexBuffer(p->vertexBufferPtr[positionStream], &size);
		for (int i = 0; i < mVertexCount; i++)
		{
			if (positionDim == 3)
				mPositions[i] = *(Vector3f*)(vb + i * attrSize + positionOffset);
			else if (positionDim == 4)
				mPositions[i] = *(Vector4f*)(vb + i * attrSize + positionOffset);
		}
		s_CurrentAPI->EndReadVertexBuffer(p->vertexBufferPtr[positionStream]);

		vb = (byte*)s_CurrentAPI->BeginReadVertexBuffer(p->vertexBufferPtr[normalStream], &size);
		for (int i = 0; i < mVertexCount; i++)
		{
			if (normalDim == 2)
			{			
				mNormals[i] = *(Vector2f*)(vb + i * attrSize + normalOffset);
				mNormals[i].z = sqrtf(1 - mNormals[i].x * mNormals[i].x - mNormals[i].y * mNormals[i].y);
			}
			else if (normalDim == 3)
				mNormals[i] = *(Vector3f*)(vb + i * attrSize + normalOffset);
			else if (normalDim == 4)
				mNormals[i] = *(Vector4f*)(vb + i * attrSize + normalOffset);
		}
		s_CurrentAPI->EndReadVertexBuffer(p->vertexBufferPtr[normalStream]);

		vb = (byte*)s_CurrentAPI->BeginReadVertexBuffer(p->vertexBufferPtr[uvStream], &size);
		for (int i = 0; i < mVertexCount; i++)
			mUV1[i] = *(Vector2f*)(vb + i * attrSize + uvOffset);
		s_CurrentAPI->EndReadVertexBuffer(p->vertexBufferPtr[uvStream]);

		byte* ib = (byte*)s_CurrentAPI->BeginReadIndexBuffer(p->indexBufferPtr, &size);
		if (p->indexFormat)
			cudaMemcpy(mIndices, ib, sizeof(int)*p->indexBufferLength, cudaMemcpyKind::cudaMemcpyHostToHost);
		else
			for (int i = 0, short* p = (short*)ib; i < mIndexCount; i++)
				mIndices[i] = p[i];
		s_CurrentAPI->EndReadIndexBuffer(p->indexBufferPtr);

		return true;
	}


	__host__ bool Mesh::CopyFromDevice(const Mesh & m)
	{
		mVertexCount = m.mVertexCount;
		mIndexCount = m.mIndexCount;

		ASSERT(cudaMallocManaged(&mPositions, sizeof(Vector3f) * mVertexCount));
		ASSERT(cudaMemcpy(mPositions, m.mPositions, sizeof(Vector3f) * mVertexCount, cudaMemcpyDeviceToDevice));

		ASSERT(cudaMallocManaged(&mNormals, sizeof(Vector3f) * mVertexCount));
		ASSERT(cudaMemcpy(mNormals, m.mNormals, sizeof(Vector3f) * mVertexCount, cudaMemcpyDeviceToDevice));

		ASSERT(cudaMallocManaged(&mUV1, sizeof(Vector2f) * mVertexCount));
		ASSERT(cudaMemcpy(mUV1, m.mUV1, sizeof(Vector2f) * mVertexCount, cudaMemcpyDeviceToDevice));

		ASSERT(cudaMallocManaged(&mIndices, sizeof(int) * mIndexCount));
		ASSERT(cudaMemcpy(mIndices, m.mIndices, sizeof(int) * mIndexCount, cudaMemcpyDeviceToDevice));

		return true;
	}
}