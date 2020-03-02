#pragma once

namespace RadGrabber
{
	struct FrameRequest;
	struct FrameInput;

	int IncrementalPTSampling(FrameRequest* hostReq, FrameInput* deviceReq);
	int IncrementalPTSamplingWithBVH(FrameRequest* hostReq, FrameInput* deviceReq);

}
