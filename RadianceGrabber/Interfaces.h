#include "Define.h"

#pragma once

namespace RadGrabber
{
	struct Ray;
	struct SurfaceIntersection;
	struct GeometryInput;

	//struct IAggregateParam
	//{
	//	virtual MeshChunk* GetStaticMeshs() const PURE;
	//	virtual int GetStaticMeshCount() const PURE;
	//	virtual MeshChunk* GetSkinnedMeshs() const PURE;
	//	virtual int GetSkinnedMeshCount() const PURE;
	//	virtual MeshRendererChunk* GetMeshRenderers() const PURE;
	//	virtual int GetMeshRendererCount() const PURE;
	//	virtual SkinnedMeshRendererChunk* GetSkinnedMeshRenderers() const PURE;
	//	virtual int GetSkinnedMeshRendererCount() const PURE;
	//	virtual LightChunk* GetLights() const PURE;
	//	virtual int GetLightCount() const PURE;
	//};

	class IAggregate
	{
	public:
		__host__ __device__ virtual void InitAggregate(const GeometryInput* param) PURE;
		__device__ virtual bool Intersect(const Ray& ray, SurfaceIntersection& isect) PURE;
	};

	class IIntegrator
	{
	public:
		__host__ virtual void Render(const IAggregate& scene) PURE;
	};
}