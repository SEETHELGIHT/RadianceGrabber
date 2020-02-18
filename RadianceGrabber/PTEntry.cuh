#pragma once

namespace RadGrabber
{
	struct UnityFrameRequest;
	struct UnityFrameInput;

	int IncrementalPTSampling(UnityFrameRequest* hostReq, UnityFrameInput* deviceReq);
	int IncrementalPTSamplingWithBVH(UnityFrameRequest* hostReq, UnityFrameInput* deviceReq);

}
