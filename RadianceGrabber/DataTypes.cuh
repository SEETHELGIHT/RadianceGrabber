#include <iostream>
#include "Define.h"

#pragma once

namespace RadGrabber
{
	__host__ float RandomFloat(float min, float max);

	typedef unsigned long ulong;
	typedef unsigned int uint;
	typedef unsigned short ushort;
	typedef unsigned char ubyte;

	template <typename Type>
	struct Vector2;
	template <typename Type>
	struct Vector3;
	template <typename Type>
	struct Vector4;

	template <typename Type>
	struct Vector2
	{
		union
		{
			struct
			{
				Type x;
				Type y;
			};
			byte data[2 * sizeof(Type)];
		};

		__forceinline__ __device__ __host__ Vector2() : x(0), y(0) { }
		__forceinline__ __device__ __host__ Vector2(Type x, Type y) : x(x), y(y) {}
		__forceinline__ __device__ __host__ Vector2(const Vector2<Type>& v) : x(v.x), y(v.y) {}

		__forceinline__ __device__ __host__ operator Vector3<Type>() const;
		__forceinline__ __device__ __host__ operator Vector4<Type>() const;

		__forceinline__ __device__ __host__ Type& operator[] (unsigned int i)
		{
			return data[i & 0x01];
		}

		__forceinline__ __device__ __host__ const Type& operator[] (unsigned int i) const
		{
			return data[i & 0x01];
		}

		__forceinline__ __device__ __host__ const Vector2<Type>& operator+() const { return *this; }
		__forceinline__ __device__ __host__ Vector2<Type> operator-() const { return Vector2<Type>(-x, -y); }

		__forceinline__ __device__ __host__ Vector2<Type>& operator+=(const Vector2<Type>& vec2)
		{
			this->x += vec2.x;
			this->y += vec2.y;
			return *this;
		}
		__forceinline__ __device__ __host__ Vector2<Type>& operator-=(const Vector2<Type>& vec2)
		{
			this->x -= vec2.x;
			this->y -= vec2.y;
			return *this;
		}
		__forceinline__ __device__ __host__ Vector2<Type>& operator*=(const float t)
		{
			this->x *= t;
			this->y *= t;
			return *this;
		}
		__forceinline__ __device__ __host__ Vector2<Type>& operator/=(const float t)
		{
			this->x /= t;
			this->y /= t;
			return *this;
		}

		__forceinline__ __device__ __host__ Type magnitude() const { return sqrt(x*x + y * y); }
		__forceinline__ __device__ __host__ Type sqrMagnitude() const { return x * x + y * y; }

		__forceinline__ __device__ __host__ Vector2<Type> normalized() const { return *this / magnitude(); }
		__forceinline__ __device__ __host__ Vector2<Type>& normalize() { return *this /= magnitude(); }

		__forceinline__ __device__ __host__ static Vector2<Type> zero() { return Vector2<Type>(0, 0); }
		__forceinline__ __device__ __host__ static Vector2<Type> one() { return Vector2<Type>(1, 1); }
	};

	template <typename Type>
	__forceinline__ __device__ __host__ Type Dot(Vector2<Type> v1, Vector2<Type> v2)
	{
		return v1.x * v2.x + v1.y * v2.y;
	}

	template <typename Type>
	__device__ __host__ Vector2<Type> Reflect(const Vector2<Type>& v, const Vector2<Type>& n)
	{
		return v - 2 * Dot<Type>(v, n) * n;
	}

	template <typename Type>
	__forceinline__ __device__ __host__ Vector2<Type> operator+(Vector2<Type> v1, Vector2<Type> v2)
	{
		return Vector2<Type>(v1.x + v2.x, v1.y + v2.y);
	}

	template <typename Type>
	__forceinline__ __device__ __host__ Vector2<Type> operator-(Vector2<Type> v1, Vector2<Type> v2)
	{
		return Vector2<Type>(v1.x - v2.x, v1.y - v2.y);
	}

	template <typename Type>
	__forceinline__ __device__ __host__ Vector2<Type> operator*(Vector2<Type> v, float t)
	{
		return Vector2<Type>(v.x * t, v.y * t);
	}

	template <typename Type>
	__forceinline__ __device__ __host__ Vector2<Type> operator*(Vector2<Type> v1, Vector2<Type> v2)
	{
		return Vector2<Type>(v1.x * v2.x, v1.y * v2.y);
	}

	template <typename Type>
	__forceinline__ __device__ __host__ Vector2<Type> operator*(float t, Vector2<Type> v)
	{
		return Vector2<Type>(v.x * t, v.y * t);
	}

	template <typename Type>
	__forceinline__ __device__ __host__ Vector2<Type> operator/(Vector2<Type> v, float t)
	{
		return Vector2<Type>(v.x / t, v.y / t);
	}

	template <typename Type>
	__forceinline__ __device__ __host__ Vector2<Type> operator/(float t, Vector2<Type> v)
	{
		return Vector2<Type>(v.x / t, v.y / t);
	}

	template <typename Type>
	__forceinline__ __device__ __host__ std::istream& operator>>(std::istream &i, Vector2<Type>& v)
	{
		i >> v.x >> v.y;
		return i;
	}

	template <typename Type>
	__forceinline__ __device__ __host__ std::ostream& operator<<(std::ostream &o, Vector2<Type>& v)
	{
		o << v.x << ' ' << v.y;
		return o;
	}

	typedef Vector2<float>				Vector2f;
	typedef Vector2<int>				Vector2i;
	typedef Vector2<unsigned int>		Vector2u;

	template <typename Type>
	struct Vector3
	{
		union
		{
			struct
			{
				Type x;
				Type y;
				Type z;
			};
			byte data[3 * sizeof(Type)];
		};

		__forceinline__ __device__ __host__ Vector3() : x(0), y(0), z(0) { }
		__forceinline__ __device__ __host__ Vector3(Type x, Type y, Type z) : x(x), y(y), z(z) {}
		__forceinline__ __device__ __host__ Vector3(const Vector3<Type>& v) : x(v.x), y(v.y), z(v.z) {}

		__forceinline__ __device__ __host__ operator Vector2<Type>() const;
		__forceinline__ __device__ __host__ operator Vector4<Type>() const;

		__forceinline__ __device__ __host__ Type& operator[] (unsigned int i)
		{
			return data[i % 3];
		}

		__forceinline__ __device__ __host__ const Type& operator[] (unsigned int i) const
		{
			return data[i % 3];
		}

		__forceinline__ __device__ __host__ const Vector3<Type>& operator+() const { return *this; }
		__forceinline__ __device__ __host__ Vector3<Type> operator-() const { return Vector3<Type>(-x, -y, -z); }

		__forceinline__ __device__ __host__ Vector3<Type>& operator+=(const Vector3<Type>& vec3)
		{
			this->x += vec3.x;
			this->y += vec3.y;
			this->z += vec3.z;
			return *this;
		}
		__forceinline__ __device__ __host__ Vector3<Type>& operator-=(const Vector3<Type>& vec3)
		{
			this->x -= vec3.x;
			this->y -= vec3.y;
			this->z -= vec3.z;
			return *this;
		}
		__forceinline__ __device__ __host__ Vector3<Type>& operator*=(const float t)
		{
			this->x *= t;
			this->y *= t;
			this->z *= t;
			return *this;
		}
		__forceinline__ __device__ __host__ Vector3<Type>& operator/=(const float t)
		{
			this->x /= t;
			this->y /= t;
			this->z /= t;
			return *this;
		}

		__forceinline__ __device__ __host__ Type magnitude() const { return sqrt(x*x + y * y + z * z); }
		__forceinline__ __device__ __host__ Type sqrMagnitude() const { return x * x + y * y + z * z; }

		__forceinline__ __device__ __host__ Vector3<Type> normalized() const { return *this / magnitude(); }
		__forceinline__ __device__ __host__ Vector3<Type>& normalize() { return *this /= magnitude(); }

		__forceinline__ __device__ __host__ static Vector3<Type> zero() { return Vector3<Type>(0, 0, 0); }
		__forceinline__ __device__ __host__ static Vector3<Type> one() { return Vector3<Type>(1, 1, 1); }
	};

	template <typename Type>
	__forceinline__ __device__ __host__ Type Dot(Vector3<Type> v1, Vector3<Type> v2)
	{
		return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
	}

	template <typename Type>
	__forceinline__ __device__ __host__ Vector3<Type> Cross(Vector3<Type> v1, Vector3<Type> v2)
	{
		return Vector3<Type>(
			v1.y * v2.z - v1.z * v2.y,
			v1.z * v2.x - v1.x * v2.z,
			v1.x * v2.y - v1.y * v2.x
			);
	}

	template <typename Type>
	__forceinline__ __device__ __host__ Vector3<Type> Reflect(const Vector3<Type>& v, const Vector3<Type>& n)
	{
		return v - 2 * Dot<Type>(v, n) * n;
	}


	template <typename Type>
	__device__ __host__ bool Refract(const Vector3<Type>& v, const Vector3<Type>& n, float ni_over_nt, Vector3<Type>& refracted)
	{
		Vector3<Type> uv = v.normalized();
		float dt = Dot(uv, n);
		float discriminant = 1.f - ni_over_nt * ni_over_nt * (1 - dt * dt);
		if (discriminant > 0)
		{
			refracted = ni_over_nt * (uv - n * dt) - n * sqrt(discriminant);
			return true;
		}
		else
			return false;
	}

	template <typename Type>
	__forceinline__ __device__ __host__ Vector3<Type> operator+(Vector3<Type> v1, Vector3<Type> v2)
	{
		return Vector3<Type>(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
	}

	template <typename Type>
	__forceinline__ __device__ __host__ Vector3<Type> operator-(Vector3<Type> v1, Vector3<Type> v2)
	{
		return Vector3<Type>(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
	}

	template <typename Type>
	__forceinline__ __device__ __host__ Vector3<Type> operator*(Vector3<Type> v, float t)
	{
		return Vector3<Type>(v.x * t, v.y * t, v.z * t);
	}

	template <typename Type>
	__forceinline__ __device__ __host__ Vector3<Type> operator*(Vector3<Type> v1, Vector3<Type> v2)
	{
		return Vector3<Type>(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
	}

	template <typename Type>
	__forceinline__ __device__ __host__ Vector3<Type> operator*(float t, Vector3<Type> v)
	{
		return Vector3<Type>(v.x * t, v.y * t, v.z * t);
	}

	template <typename Type>
	__forceinline__ __device__ __host__ Vector3<Type> operator/(Vector3<Type> v, float t)
	{
		return Vector3<Type>(v.x / t, v.y / t, v.z / t);
	}

	template <typename Type>
	__forceinline__ __device__ __host__ Vector3<Type> operator/(float t, Vector3<Type> v)
	{
		return Vector3<Type>(v.x / t, v.y / t, v.z / t);
	}

	template <typename Type>
	__forceinline__ __device__ __host__ std::istream& operator>>(std::istream &i, Vector3<Type>& v)
	{
		i >> v.x >> v.y >> v.z;
		return i;
	}

	template <typename Type>
	__forceinline__ __device__ __host__ std::ostream& operator<<(std::ostream &o, Vector3<Type>& v)
	{
		o << v.x << ' ' << v.y << ' ' << v.z;
		return o;
	}

#define MachineEpsilon (std::numeric_limits<float>::epsilon() * 0.5)

#ifndef MAX
#define MAX(a,b)            (((a) > (b)) ? (a) : (b))
#endif

#ifndef MIN
#define MIN(a,b)            (((a) < (b)) ? (a) : (b))
#endif

	template <typename T>
	__forceinline__ __host__ __device__ int MaxDimension(const Vector3<T> &v) {
		return (v.x > v.y) ? ((v.x > v.z) ? 0 : 2) : ((v.y > v.z) ? 1 : 2);
	}

	template <typename T>
	__forceinline__ __host__ __device__ Vector3<T> Min(const Vector3<T> &p1, const Vector3<T> &p2) {
		return Vector3<T>(MIN(p1.x, p2.x), MIN(p1.y, p2.y), MIN(p1.z, p2.z));
	}

	template <typename T>
	__forceinline__ __host__ __device__ Vector3<T> Max(const Vector3<T> &p1, const Vector3<T> &p2) {
		return Vector3<T>(MAX(p1.x, p2.x), MAX(p1.y, p2.y), MAX(p1.z, p2.z));
	}

	template <typename T>
	__forceinline__ __host__ __device__ Vector3<T> Abs(const Vector3<T> &v) {
		return Vector3<T>(std::abs(v.x), std::abs(v.y), std::abs(v.z));
	}

	template <typename T>
	__forceinline__ __host__ __device__ Vector3<T> Permute(const Vector3<T> &v, int x, int y, int z) {
		return Vector3<T>(v[x], v[y], v[z]);
	}

	template <typename T>
	__forceinline__ __host__ __device__ T MaxComponent(const Vector3<T> &v) {
		return MAX(v.x, MAX(v.y, v.z));
	}

	template <>
	__forceinline__ __host__ __device__ Vector3<float> Min(const Vector3<float> &p1, const Vector3<float> &p2) {
#ifndef __CUDA_ARCH__
		return Vector3<float>(MIN(p1.x, p2.x), MIN(p1.y, p2.y), MIN(p1.z, p2.z));
#else //#ifdef __CUDA_ARCH__
		return Vector3<float>(fminf(p1.x, p2.x), fminf(p1.y, p2.y), fminf(p1.z, p2.z));
#endif
	}

	template <>
	__forceinline__ __host__ __device__ Vector3<float> Max(const Vector3<float> &p1, const Vector3<float> &p2) {
#ifndef __CUDA_ARCH__
		return Vector3<float>(MAX(p1.x, p2.x), MAX(p1.y, p2.y), MAX(p1.z, p2.z));
#else //#ifdef __CUDA_ARCH__
		return Vector3<float>(fmaxf(p1.x, p2.x), fmaxf(p1.y, p2.y), fmaxf(p1.z, p2.z));
#endif
	}

	template <>
	__forceinline__ __host__ __device__ Vector3<float> Abs(const Vector3<float> &v) {
#ifndef __CUDA_ARCH__
		return Vector3<float>(std::abs(v.x), std::abs(v.y), std::abs(v.z));
#else //#ifdef __CUDA_ARCH__
		return Vector3<float>(fabsf(v.x), fabsf(v.y), fabsf(v.z));
#endif
	}

	template <>
	__forceinline__ __host__ __device__ float MaxComponent(const Vector3<float> &v) {
#ifndef __CUDA_ARCH__
		return MAX(v.x, MAX(v.y, v.z));
#else //#ifdef __CUDA_ARCH__
		return fmaxf(v.x, fmaxf(v.y, v.z));
#endif
	}

	typedef Vector3<float>				Vector3f;
	typedef Vector3<int>				Vector3i;
	typedef Vector3<unsigned int>		Vector3u;

	template <typename Type>
	struct Vector4
	{
		union
		{
			struct
			{
				Type x;
				Type y;
				Type z;
				Type w;
			};
			byte data[4 * sizeof(Type)];
		};

		__forceinline__ __device__ __host__ Vector4() : x(0), y(0), z(0), w(0) { }
		__forceinline__ __device__ __host__ Vector4(Type x, Type y, Type z, Type w) : x(x), y(y), z(z), w(w) {}
		__forceinline__ __device__ __host__ Vector4(const Vector4<Type>& v) : x(v.x), y(v.y), z(v.z), w(v.w) {}

		__forceinline__ __device__ __host__ operator Vector2<Type>() const;
		__forceinline__ __device__ __host__ operator Vector3<Type>() const;

		__forceinline__ __device__ __host__ Type& operator[] (unsigned int i)
		{
			return data[i & 0x03];
		}

		__forceinline__ __device__ __host__ const Type& operator[] (unsigned int i) const
		{
			return data[i & 0x03];
		}

		__forceinline__ __device__ __host__ const Vector4<Type>& operator+() const { return *this; }
		__forceinline__ __device__ __host__ Vector4<Type> operator-() const { return Vector4<Type>(-x, -y, -z, -w); }

		__forceinline__ __device__ __host__ Vector4<Type>& operator+=(const Vector4<Type>& vec3)
		{
			this->x += vec3.x;
			this->y += vec3.y;
			this->z += vec3.z;
			this->w += vec3.w;
			return *this;
		}
		__forceinline__ __device__ __host__ Vector4<Type>& operator-=(const Vector4<Type>& vec3)
		{
			this->x -= vec3.x;
			this->y -= vec3.y;
			this->z -= vec3.z;
			this->w -= vec3.w;
			return *this;
		}
		__forceinline__ __device__ __host__ Vector4<Type>& operator*=(const float t)
		{
			this->x *= t;
			this->y *= t;
			this->z *= t;
			this->w *= t;
			return *this;
		}
		__forceinline__ __device__ __host__ Vector4<Type>& operator/=(const float t)
		{
			this->x /= t;
			this->y /= t;
			this->z /= t;
			this->w /= t;
			return *this;
		}

		__forceinline__ __device__ __host__ Type magnitude() const { return sqrt(x*x + y * y + z * z + w * w); }
		__forceinline__ __device__ __host__ Type sqrMagnitude() const { return x * x + y * y + z * z + w * w; }

		__forceinline__ __device__ __host__ Vector4<Type> normalized() const { return *this / magnitude(); }
		__forceinline__ __device__ __host__ Vector4<Type>& normalize() { return *this /= magnitude(); }

		__forceinline__ __device__ __host__ static Vector4<Type> zero() { return Vector4<Type>(0, 0, 0, 0); }
		__forceinline__ __device__ __host__ static Vector4<Type> one() { return Vector4<Type>(1, 1, 1, 1); }
	};

	template <typename Type>
	__forceinline__ __device__ __host__ Type Dot(Vector4<Type> v1, Vector4<Type> v2)
	{
		return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w;
	}

	template <typename Type>
	__device__ __host__ Vector4<Type> Reflect(const Vector4<Type>& v, const Vector4<Type>& n)
	{
		return v - 2 * Dot<Type>(v, n) * n;
	}

	template <typename Type>
	__forceinline__ __device__ __host__ Vector4<Type> operator+(Vector4<Type> v1, Vector4<Type> v2)
	{
		return Vector4<Type>(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z, v1.w + v2.w);
	}

	template <typename Type>
	__forceinline__ __device__ __host__ Vector4<Type> operator-(Vector4<Type> v1, Vector4<Type> v2)
	{
		return Vector4<Type>(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z, v1.z - v2.z);
	}

	template <typename Type>
	__forceinline__ __device__ __host__ Vector4<Type> operator*(Vector4<Type> v, float t)
	{
		return Vector4<Type>(v.x * t, v.y * t, v.z * t, v.w * t);
	}

	template <typename Type>
	__forceinline__ __device__ __host__ Vector4<Type> operator*(Vector4<Type> v1, Vector4<Type> v2)
	{
		return Vector4<Type>(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z, v1.w * v2.w);
	}

	template <typename Type>
	__forceinline__ __device__ __host__ Vector4<Type> operator*(float t, Vector4<Type> v)
	{
		return Vector4<Type>(v.x * t, v.y * t, v.z * t, v.w * t);
	}

	template <typename Type>
	__forceinline__ __device__ __host__ Vector4<Type> operator/(Vector4<Type> v, float t)
	{
		return Vector4<Type>(v.x / t, v.y / t, v.z / t, v.w / t);
	}

	template <typename Type>
	__forceinline__ __device__ __host__ Vector4<Type> operator/(float t, Vector4<Type> v)
	{
		return Vector4<Type>(v.x / t, v.y / t, v.z / t, v.w / t);
	}

	template <typename Type>
	__forceinline__ __device__ __host__ std::istream& operator>>(std::istream &i, Vector4<Type>& v)
	{
		i >> v.x >> v.y >> v.z >> v.w;
		return i;
	}

	template <typename Type>
	__forceinline__ __device__ __host__ std::ostream& operator<<(std::ostream &o, Vector4<Type>& v)
	{
		o << v.x << ' ' << v.y << ' ' << v.z << ' ' << v.w;
		return o;
	}

	typedef Vector4<float>				Vector4f;
	typedef Vector4<int>				Vector4i;
	typedef Vector4<unsigned int>		Vector4u;

	template <typename Type>
	__forceinline__ __device__ __host__ Vector2<Type>::operator Vector3<Type>() const
	{
		return Vector3<Type>(this->x, this->y, 0);
	}

	template <typename Type>
	__forceinline__ __device__ __host__ Vector2<Type>::operator Vector4<Type>() const
	{
		return Vector4<Type>(this->x, this->y, 0, 0);
	}

	template <typename Type>
	__forceinline__ __device__ __host__ Vector3<Type>::operator Vector2<Type>() const
	{
		return Vector2<Type>(this->x, this->y);
	}

	template <typename Type>
	__forceinline__ __device__ __host__ Vector3<Type>::operator Vector4<Type>() const
	{
		return Vector4<Type>(this->x, this->y, this->z, 0);
	}

	template <typename Type>
	__forceinline__ __device__ __host__ Vector4<Type>::operator Vector2<Type>() const
	{
		return Vector2<Type>(this->x, this->y);
	}

	template <typename Type>
	__forceinline__ __device__ __host__ Vector4<Type>::operator Vector3<Type>() const
	{
		return Vector3<Type>(this->x, this->y, this->z);
	}

	struct Quaternion
	{
		union
		{
			struct
			{
				float x;
				float y;
				float z;
				float w;
			};
			char data[16];
		};

		__forceinline__ __device__ __host__ Quaternion() { }
		__forceinline__ __device__ __host__ Quaternion(float _x, float _y, float _z, float _w) : x(_x), y(_y), z(_z), w(_w) {  }
		__forceinline__ __device__ __host__ Quaternion(const Quaternion& q) : x(q.x), y(q.y), z(q.z), w(q.w) { }

		//__forceinline__ __device__ __host__ Quaternion operator*(const Quaternion& q)
		//{
		//	return Quaternion(
		//		w*q.x + x * q.w + y * q.z - z * q.y,
		//		w*q.y + y * q.w + z * q.x - x * q.z,
		//		w*q.z + z * q.w + x * q.y - y * q.x,
		//		w*q.w - x * q.x - y * q.y - z * q.z);
		//}

		__forceinline__ __device__ __host__ Quaternion operator/(const float f)
		{
			ASSERT(f != 0);
			return Quaternion(x / f, y / f, z / f, w / f);
		}

		__forceinline__ __device__ __host__ float Magnitude() const
		{
			return sqrt(x*x + y*y + z*z + w*w);
		}

		__forceinline__ __device__ __host__ operator Vector3f() const
		{
			return Vector3f(x, y, z);
		}
	};

	__forceinline__ __device__ __host__ void Inverse(Quaternion& q)
	{
		q.x = -q.x;
		q.y = -q.y;
		q.z = -q.z;
	}

	__forceinline__ __device__ __host__ Quaternion operator*(const Quaternion& q1, const Quaternion& q2)
	{
		return Quaternion(
			q2.w*q1.x + q2.x * q1.w + q2.y * q1.z - q2.z * q1.y,
			q2.w*q1.y + q2.y * q1.w + q2.z * q1.x - q2.x * q1.z,
			q2.w*q1.z + q2.z * q1.w + q2.x * q1.y - q2.y * q1.x,
			q2.w*q1.w - q2.x * q1.x - q2.y * q1.y - q2.z * q1.z);
	}

	__forceinline__ __device__ __host__ void Rotate(const Quaternion& q1, INOUT Vector3f& v)
	{
		// implication : quaternion is normalized.
		Quaternion p(v.x, v.y, v.z, 0), q2(-q1.x, -q1.y, -q1.z, q1.w), q;

		q = q1 * p;
		q = q * q2;
		v.x = q.x;
		v.y = q.y;
		v.z = q.z;
	}

	struct ColorRGB
	{
		union
		{
			struct
			{
				float r;
				float g;
				float b;
			};
			char data[12];
		};

		__forceinline__ __device__ __host__ ColorRGB() : r(0), g(0), b(0) {}
		__forceinline__ __device__ __host__ ColorRGB(float r, float g, float b) : r(r), g(g), b(b) {}
		__forceinline__ __device__ __host__ ColorRGB(const ColorRGB& c) : r(c.r), g(c.g), b(c.b) {}
	};

	struct ColorRGBA
	{
		union
		{
			struct
			{
				float r;
				float g;
				float b;
				float a;
			};
			char data[16];
		};

		__forceinline__ __device__ __host__ ColorRGBA() : r(0), g(0), b(0), a(0) {}
		__forceinline__ __device__ __host__ ColorRGBA(float r, float g, float b, float a) : r(r), g(g), b(b), a(a) {}
		__forceinline__ __device__ __host__ ColorRGBA(const ColorRGBA& c) : r(c.r), g(c.g), b(c.b), a(c.a) {}
	};

	struct ColorRGBA32
	{
		union
		{
			struct
			{
				byte r;
				byte g;
				byte b;
				byte a;
			};
			char data[4];
		};

		__forceinline__ __device__ __host__ ColorRGBA32() : r(0), g(0), b(0), a(0) {}
		__forceinline__ __device__ __host__ ColorRGBA32(byte r, byte g, byte b, byte a) : r(r), g(g), b(b), a(a) {}
		__forceinline__ __device__ __host__ ColorRGBA32(const ColorRGBA32& c) : r(c.r), g(c.g), b(c.b), a(c.a) {}
	};

	/*
		- column-major matrix 4x4 in Unity
		https://en.wikipedia.org/wiki/Row-_and_column-major_order
	*/
	struct Matrix4x4
	{
		union
		{
			struct
			{
				float m00;
				float m10;
				float m20;
				float m30;

				float m01;
				float m11;
				float m21;
				float m31;

				float m02;
				float m12;
				float m22;
				float m32;

				float m03;
				float m13;
				float m23;
				float m33;
			};
			struct
			{
				Vector4f v0;
				Vector4f v1;
				Vector4f v2;
				Vector4f v3;
			};
			char data[96];
		};

		__forceinline__ __device__ __host__ Matrix4x4()
		{
			m00 = m10 = m20 = m30 = m01 = m11 = m12 = m13 = m02 = m12 = m22 = m32 = m03 = m13 = m23 = m33 = 0;
		}
		__forceinline__ __device__ __host__ Matrix4x4(const Matrix4x4& mat)
		{
			memcpy(data, mat.data, sizeof(Matrix4x4)); 
		}
		__forceinline__ __device__ __host__ Matrix4x4(const Vector4f& v0, const Vector4f& v1, const Vector4f& v2, const Vector4f& v3)
		{
			this->v0 = v0;
			this->v1 = v1;
			this->v2 = v2;
			this->v3 = v3;
		}

		__forceinline__ __device__ __host__ static Matrix4x4 GetIdentity()
		{
			Matrix4x4 m;
			m.m00 = m.m11 = m.m22 = m.m33 = 1;
			return m;
		}
		__forceinline__ __device__ __host__ static Matrix4x4 FromTRS(Vector3f translate, Quaternion rotation, Vector3f scale)
		{
			Matrix4x4 m;
			return m;
		}
	};

	struct Ray
	{
		union
		{
			struct
			{
				Vector3f origin;
				Vector3f direction;
			};
			byte data[24];
		};

		__forceinline__ __device__ __host__ Ray() { }
		__forceinline__ __device__ __host__ Ray(Vector3f o, Vector3f d) { origin = o; direction = d; }
		__forceinline__ __device__ __host__ Ray(const Ray& r) : origin(r.origin), direction(r.direction) { }
	};

	struct Bounds
	{
		union
		{
			struct
			{
				Vector3f				center;
				Vector3f				size;
			};
			byte data[24];
		};

		__forceinline__ __device__ __host__ Bounds() { }
		__forceinline__ __device__ __host__ Bounds(const Bounds& b) : center(b.center), size(b.size) { }

		__forceinline__ __device__ __host__ bool Intersect(const Ray& r)
		{
			// TODO:: Intersect. Bound vs Ray 

			return false;
		}
	};

	struct SurfaceIntersection
	{
		Vector3f position;
		Vector3f normal;
		Vector3f tangent;
		Vector2f uv;
		int emitted : 1;
		int materialIndex : 31;
		int skinnedRenderer : 1;
		int rendererIndex : 31;
	};

}
