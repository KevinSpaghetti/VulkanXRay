#version 450

layout(location = 0) out vec4 frag_color;

layout(location = 0) in vec3 ws_position;
layout(location = 1) in vec3 normal;

const vec3 light_wp = {5.0f, 5.0f, 5.0f};

void main() {
    vec3 light_dir = light_wp - ws_position;
    float light = clamp(dot(normalize(light_dir), normalize(normal)), 0., 1.);
    vec3 diffuse_color = { 0.75f, 0.75f, 0.75f };
    frag_color = vec4(diffuse_color * light, 1.0);
}
