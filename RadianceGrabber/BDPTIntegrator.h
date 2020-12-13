#include "DataTypes.cuh"
#include "Integrator.h"
#include "Marshal.cuh"

#pragma once

namespace RadGrabber
{
	class BDPTIntegrator : public ICancelableIntergrator, public ICancelableIterativeIntergrator
	{
	public:
		BDPTIntegrator();
		~BDPTIntegrator();

		// ICancelableIntergrator��(��) ���� ��ӵ�
		__host__ virtual void RenderStraight(const IAggregate * getDeviceScene(int mutableIndex), const HostDevicePair<IMultipleInput*>& in, IColorTarget & target, const RequestOption & opt, OptimalLaunchParam & param) override;
		__host__ virtual void RenderIncremental(const IAggregate * getDeviceScene(int mutableIndex), const HostDevicePair<IMultipleInput*>& in, IColorTarget & target, const RequestOption & opt, OptimalLaunchParam & param) override;

		// ICancelableIterativeIntergrator��(��) ���� ��ӵ�
		__host__ virtual void RenderStraight(const IIteratableAggregate * getDeviceScene(int mutableIndex), const HostDevicePair<IMultipleInput*>& in, IColorTarget & target, const RequestOption & opt, OptimalLaunchParam & param) override;
		__host__ virtual void RenderIncremental(const IIteratableAggregate * getDeviceScene(int mutableIndex), const HostDevicePair<IMultipleInput*>& in, IColorTarget & target, const RequestOption & opt, OptimalLaunchParam & param) override;

	public:
		__host__ virtual void ReserveCancel() override;
		__host__ virtual bool IsCancel() override;
	};
}