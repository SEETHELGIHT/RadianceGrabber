#include <chrono>
#include <ratio>
#include <vector>
#include <cuda.h>
#include <device_functions.h>
#include <cooperative_groups.h>

using namespace cooperative_groups;

#include "Integrator.h"
#include "ColorTarget.h"
#include "Aggregate.h"
#include "Marshal.cuh"
#include "DeviceConfig.h"
#include "Util.h"

namespace RadGrabber
{
	/*
		TODO:: MLT 구현
	*/
	/*
		4. russian roulette 으로 중간에 멈추기 처리?
			구현 이후에
		5. Subsurface Scattering, Transmission 구현은 나중에 ㅎㅎ
	*/

}
