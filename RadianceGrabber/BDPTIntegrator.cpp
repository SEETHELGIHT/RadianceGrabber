#include "BDPTIntegrator.h"


namespace RadGrabber
{
	BDPTIntegrator::BDPTIntegrator()
	{
	}


	BDPTIntegrator::~BDPTIntegrator()
	{
	}

	__host__ void RadGrabber::BDPTIntegrator::ReserveCancel()
	{
	}

	__host__ bool RadGrabber::BDPTIntegrator::IsCancel()
	{
		return false;
	}

	__host__ void RadGrabber::BDPTIntegrator::RenderStraight(const IAggregate * getDeviceScene(int mutableIndex), const HostDevicePair<IMultipleInput*>& in, IColorTarget & target, const RequestOption & opt, OptimalLaunchParam & param)
	{
	}

	__host__ void RadGrabber::BDPTIntegrator::RenderIncremental(const IAggregate * getDeviceScene(int mutableIndex), const HostDevicePair<IMultipleInput*>& in, IColorTarget & target, const RequestOption & opt, OptimalLaunchParam & param)
	{
	}

	__host__ void RadGrabber::BDPTIntegrator::RenderStraight(const IIteratableAggregate * getDeviceScene(int mutableIndex), const HostDevicePair<IMultipleInput*>& in, IColorTarget & target, const RequestOption & opt, OptimalLaunchParam & param)
	{
	}

	__host__ void RadGrabber::BDPTIntegrator::RenderIncremental(const IIteratableAggregate * getDeviceScene(int mutableIndex), const HostDevicePair<IMultipleInput*>& in, IColorTarget & target, const RequestOption & opt, OptimalLaunchParam & param)
	{
	}
}
