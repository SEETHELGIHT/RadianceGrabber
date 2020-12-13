#include "DataTypes.cuh"

#pragma once

namespace RadGrabber
{
	//struct FrameConstant
	//{
	//	int currentSamplingCount;
	//	union
	//	{
	//		struct
	//		{
	//			float cameraProjectionInverseArray[16];
	//			float cameraSpaceInverseArray[16];
	//		};
	//		struct
	//		{
	//			Matrix4x4 cameraProjectionInverseMatrix;
	//			Matrix4x4 cameraSpaceInverseMatrix;
	//		};
	//	};

	//	__forceinline__ __host__ __device__ FrameConstant() {}
	//};

	//__host__ void SetFrameConstant(const FrameConstant* c);
	//__host__ __device__ const FrameConstant& GetFrameConstant();

	class IAggregate;
	class IIteratableAggregate;
	class IMultipleInput;
	class IColorTarget;
	struct RequestOption;
	struct OptimalLaunchParam;

	class IIntegrator abstract
	{
	public:
		__host__ virtual void RenderStraight(const IAggregate* getDeviceScene(int mutableIndex), const HostDevicePair<IMultipleInput*>& in, IColorTarget& target, const RequestOption& opt, OptimalLaunchParam& param) PURE;
		__host__ virtual void RenderIncremental(const IAggregate* getDeviceScene(int mutableIndex), const HostDevicePair<IMultipleInput*>& in, IColorTarget& target, const RequestOption& opt, OptimalLaunchParam& param) PURE;
	};

	class IIterativeIntegrator abstract
	{
	public:
		__host__ virtual void RenderStraight(const IIteratableAggregate* getDeviceScene(int mutableIndex), const HostDevicePair<IMultipleInput*>& in, IColorTarget& target, const RequestOption& opt, OptimalLaunchParam& param) PURE;
		__host__ virtual void RenderIncremental(const IIteratableAggregate* getDeviceScene(int mutableIndex), const HostDevicePair<IMultipleInput*>& in, IColorTarget& target, const RequestOption& opt, OptimalLaunchParam& param) PURE;
	};

	class ICancelable abstract
	{
	public:
		__host__ virtual void ReserveCancel() PURE;
		__host__ virtual bool IsCancel() PURE;
	};

	class ICancelableIntergrator : public ICancelable, public IIntegrator { };
	class ICancelableIterativeIntergrator : public ICancelable, public IIterativeIntegrator { };
}
