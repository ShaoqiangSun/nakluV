// Shared Gerstner-style wave stack for caustic.vert and objects.frag (water surface).
#ifndef WATER_WAVES_GLSL
#define WATER_WAVES_GLSL

#define WATER_MAX_WAVES 4

void water_waves_disp_and_normal(
	vec2 local_xy,
	float time,
	vec4 wave_count,
	vec4 waves[WATER_MAX_WAVES],
	out vec3 disp_out,
	out vec3 normal_local_out)
{
	vec3 disp = vec3(0.0);
	vec2 grad = vec2(0.0);

	int count = int(wave_count.x + 0.5);
	for (int i = 0; i < count && i < WATER_MAX_WAVES; ++i) {
		float a = waves[i].x;
		float f = waves[i].y;
		float d = waves[i].z;
		float s = waves[i].w;

		vec2 dir = vec2(cos(d), sin(d));
		float phase = f * dot(dir, local_xy) + s * time;

		float c = cos(phase);
		float sn = sin(phase);

		disp.xy += dir * (-a * sn);
		disp.z  += a * c;

		grad += dir * (-a * f * sn);
	}

	normal_local_out = normalize(vec3(-grad.x, -grad.y, 1.0));
	disp_out = disp;
}

#endif
