#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 normal;

layout( push_constant ) uniform constants{
    mat4 projection_matrix;
    mat4 model_matrix;
} push_constants;

layout(location = 0) out vec3 out_position;
layout(location = 1) out vec3 out_normal;

void main() {
    vec4 ws_position = push_constants.model_matrix * vec4(position, 1.0);
    gl_Position = push_constants.projection_matrix * ws_position;

    out_normal = mat3(transpose(inverse(push_constants.model_matrix))) * normal;
    out_position = ws_position.xyz;
}