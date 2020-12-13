#pragma once
#ifndef GEOMETRY_H
#define GEOMETRY_H
#include <cmath>
#include <cstring>

typedef float Point3D[3];
typedef float Point2D[2];

template<class T>
inline void set(Point3D a, T x, T y, T z) {
	a[0] = x; a[1] = y; a[2] = z;
}

inline void set(Point3D a, Point3D p) {
	memmove(a, p, sizeof(Point3D));
}

template<class T>
inline T min3(T a, T b, T c) {
	return a<b? (a<c ? a : c) : b<c ? b : c;
}

template<class T>
inline T max3(T a, T b, T c) {
	return a>b? (a>c ? a : c) : b>c ? b : c;
}

inline void sub(Point3D a, Point3D b, Point3D c) {
        c[0] = a[0]-b[0];
        c[1] = a[1]-b[1];
        c[2] = a[2]-b[2];
}

inline void neg(Point3D a, Point3D b) {
        b[0] = -a[0];
        b[1] = -a[1];
        b[2] = -a[2];
}

inline void add(Point3D a, Point3D b, Point3D c) {
        c[0] = a[0]+b[0];
        c[1] = a[1]+b[1];
        c[2] = a[2]+b[2];
}

template<class T>
inline void div(Point3D a, T b, Point3D c) {
        c[0] = T(a[0]/b);
        c[1] = T(a[1]/b);
        c[2] = T(a[2]/b);
}

template<class T>
inline void times(Point3D a, T b, Point3D c) {
        c[0] = a[0]*b;
        c[1] = a[1]*b;
        c[2] = a[2]*b;
}

inline float dis(Point3D a) {
    return sqrt(a[0]*a[0] + a[1]*a[1] + a[2]*a[2]);
}

inline void norm(Point3D a) {
    float b = dis(a);
    div(a, b, a);
}

inline void multi(Point3D a, Point3D b, Point3D c) {
    c[2] = a[0]*b[1] - a[1]*b[0];
    c[0] = a[1]*b[2] - a[2]*b[1];
    c[1] = a[2]*b[0] - a[0]*b[2];
}

inline void multi(Point3D c, Point3D a, Point3D b, Point3D d) {
    d[2] = (a[0]-c[0])*(b[1]-c[1]) - (a[1]-c[1])*(b[0]-c[0]);
    d[0] = (a[1]-c[1])*(b[2]-c[2]) - (a[2]-c[2])*(b[1]-c[1]);
    d[1] = (a[2]-c[2])*(b[0]-c[0]) - (a[0]-c[0])*(b[2]-c[2]);
}

inline double dot(Point3D a, Point3D b) {
	return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
}

inline double dot2D(Point2D a, Point2D b) {
	return a[0]*b[0] + a[1]*b[1];
}

#endif
