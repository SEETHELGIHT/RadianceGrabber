#pragma once

#include "BinaryLayout.h"
#include "host_defines.h"

namespace RadGrabber
{
	class Mesh
	{
	public:
		__host__ Mesh();
		__host__ Mesh(const Mesh& m);
		__host__ Mesh(const Mesh&& m);
		__host__ Mesh(const MeshChunk* p);
		__host__ ~Mesh();

		int GetVertexCount();
		const Vector3f* GetPosition();
		const Vector3f* GetNormal();
		const Vector2f* GetUV();

		int GetIndexCount();
		const int* GetIndices();

	protected:
		__host__ bool CopyFromDevice(const Mesh& m);
		__host__ bool BuildFromChunk(const MeshChunk* p);

	private:
		__host__ __device__ int mVertexCount;
		__host__ __device__ Vector3f* mPositions;
		__host__ __device__ Vector3f* mNormals;
		__host__ __device__ Vector2f* mUV1;

		__host__ __device__ int mIndexCount;
		__host__ __device__ int* mIndices;
	};
}
