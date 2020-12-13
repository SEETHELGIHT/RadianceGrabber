#include "MLTIntegrator.h"

#include <chrono>
#include <ratio>
#include <vector>
#include <cuda.h>
#include <device_functions.h>
#include <cooperative_groups.h>

using namespace cooperative_groups;

#include "ColorTarget.h"
#include "Aggregate.h"
#include "Marshal.cuh"
#include "DeviceConfig.h"
#include "Util.h"

namespace RadGrabber
{
	__host__ void MLTIntergrator::RenderStraight(const IAggregate* getDeviceScene(int mutableIndex), const HostDevicePair<IMultipleInput*>& in, IColorTarget& target, const RequestOption& opt, OptimalLaunchParam& param)
	{
	}
	__host__ void MLTIntergrator::RenderIncremental(const IAggregate* getDeviceScene(int mutableIndex), const HostDevicePair<IMultipleInput*>& in, IColorTarget& target, const RequestOption& opt, OptimalLaunchParam& param)
	{
	}
	__host__ void MLTIntergrator::ReserveCancel()
	{
		reserveCancel = true;
	}
	__host__ bool MLTIntergrator::IsCancel()
	{
		return !reserveCancel;
	}


	MLTIntergrator* CreateMLTIntegrator(const RequestOption& opt) { return nullptr; }
}