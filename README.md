# RadianceGrabber

- path-tracer, work in Unity as plugin
- implemeted by CUDA and C++

# Path-Tracer

- start: using naive algorithms
	- BVH by SAH, stochastic path tracing
- may experiment several algorithms
	- dual-split trees
	- compressed wide BVH
	- metropolis light transport(+MMLT, RJMLT)

# Support at Unity

- compatible render component : MeshRenderer, SkinnedMeshRenderer
- mode: single image, record video
- mesh compression must be unchecked.

