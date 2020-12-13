#include "Marshal.cuh"

namespace RadGrabber
{
	namespace Test
	{

		/*
			implicate that hierarchical structure retain more size
		*/
		int MarshalTest()
		{
			printf("FrameInput::%ld, FrameInputInternal::%ld\n", sizeof(FrameInput), sizeof(FrameInputInternal));
			printf("FrameRequest::%ld, FrameRequestMarshal::%ld\n", sizeof(FrameRequest), sizeof(FrameRequestMarshal));
			printf("SubmeshDescriptor::%ld\n", sizeof(UnitySubMeshDescriptor));

			return 0;
		}
	}
}