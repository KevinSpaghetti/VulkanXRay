#include <iostream>

#include "libs/imgui/imgui.h"
#include "libs/imgui/backends/imgui_impl_glfw.h"
#include "libs/imgui/backends/imgui_impl_vulkan.h"

#define TINYOBJLOADER_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "libs/tinyobjloader/tiny_obj_loader.h"
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtx/transform.hpp>
#include <set>
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>
#define VULKAN_HPP_NO_CONSTRUCTORS
#include <vulkan/vulkan.hpp>
#include <fstream>

uint32_t findMemoryType(vk::PhysicalDevice pdevice, uint32_t type_filter, vk::MemoryPropertyFlags properties) {
    vk::PhysicalDeviceMemoryProperties memProperties = pdevice.getMemoryProperties();
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("failed to find suitable memory type!");
}

vk::SampleCountFlagBits get_max_msaa_samples(vk::PhysicalDeviceProperties pdevice_props){
    vk::SampleCountFlags supported_counts = pdevice_props.limits.framebufferColorSampleCounts & pdevice_props.limits.framebufferDepthSampleCounts;
    if (supported_counts & vk::SampleCountFlagBits::e64) return vk::SampleCountFlagBits::e64;
    if (supported_counts & vk::SampleCountFlagBits::e32) return vk::SampleCountFlagBits::e32;
    if (supported_counts & vk::SampleCountFlagBits::e16) return vk::SampleCountFlagBits::e16;
    if (supported_counts & vk::SampleCountFlagBits::e8) return vk::SampleCountFlagBits::e8;
    if (supported_counts & vk::SampleCountFlagBits::e4) return vk::SampleCountFlagBits::e4;
    return vk::SampleCountFlagBits::e1;
}

void cameraControls(GLFWwindow* window, glm::mat4& model){
    if (glfwGetKey(window, GLFW_KEY_SPACE)) {
        model = glm::mat4(1.0f);
        model = glm::translate(model, glm::vec3{0.0f, 0.0f, -5.0f});
    };
    if (glfwGetKey(window, GLFW_KEY_W)) {
        model = glm::translate(model, glm::vec3{0.0f, 0.0f, 1.0f});
    }
    if (glfwGetKey(window, GLFW_KEY_A)) {
        model = glm::translate(model, glm::vec3{1.0f, 0.0f, 0.0f});
    }
    if (glfwGetKey(window, GLFW_KEY_S)) {
        model = glm::translate(model, glm::vec3{0.0f, 0.0f, -1.0f});
    }
    if (glfwGetKey(window, GLFW_KEY_D)) {
        model = glm::translate(model, glm::vec3{-1.0f, 0.0f, 0.0f});
    }
    if (glfwGetKey(window, GLFW_KEY_Z)) {
        model = glm::rotate(model, glm::radians(-0.5f), glm::vec3{0.0f, 1.0f, 0.0f});
    }
    if (glfwGetKey(window, GLFW_KEY_C)) {
        model = glm::rotate(model, glm::radians(+0.5f), glm::vec3{0.0f, 1.0f, 0.0f});
    }

}

int main() {

    std::string inputfile = "resources/models/monke.obj";
    tinyobj::ObjReaderConfig reader_config;
    reader_config.mtl_search_path = "resources/models/"; // Path to material files
    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(inputfile, reader_config)) {
        if (!reader.Error().empty()) {
            std::cerr << "TinyObjReader: " << reader.Error();
        }
        exit(1);
    }
    if (!reader.Warning().empty()) {
        std::cout << "TinyObjReader: " << reader.Warning();
    }

    auto& attrib = reader.GetAttrib();
    auto& shapes = reader.GetShapes();
    auto& materials = reader.GetMaterials();


    glfwInit();

    const uint32_t WIDTH = 800;
    const uint32_t HEIGHT = 600;

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

    GLFWwindow *window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);

    vk::Instance instance;
    vk::PhysicalDevice pdevice;
    vk::Device device;

    vk::SwapchainKHR swapchain;

    const std::string appName = "Renderer";
    const std::string engineName = "Vulkan renderer";
    const uint32_t version = VK_MAKE_VERSION(1, 0, 0);

    vk::ApplicationInfo application_info{
            .pApplicationName = appName.c_str(),
            .applicationVersion = version,
            .pEngineName = engineName.c_str(),
            .apiVersion = version
    };

    uint32_t windowSystemExtensionCount = 0;
    const char **windowSystemExtensions = glfwGetRequiredInstanceExtensions(&windowSystemExtensionCount);

    std::vector<const char*> validation_layers = { "VK_LAYER_KHRONOS_validation" };
    vk::InstanceCreateInfo instance_ci{
            .pApplicationInfo = &application_info,
            .enabledLayerCount = 0,
            .ppEnabledLayerNames = nullptr,
            .enabledExtensionCount = windowSystemExtensionCount,
            .ppEnabledExtensionNames = windowSystemExtensions,
    };

    std::cout << "Enabled " + std::to_string(windowSystemExtensionCount) + " extensions for window system\n";
    for (int i = 0; i < windowSystemExtensionCount; ++i) {
        std::cout << "\t" + std::string(windowSystemExtensions[i]) + "\n";
    }

    vk::createInstance(&instance_ci, nullptr, &instance);

    auto physical_devices = instance.enumeratePhysicalDevices();
    std::cout << physical_devices.size() << std::endl;
    pdevice = physical_devices.front();

    auto queue_families = pdevice.getQueueFamilyProperties();
    uint32_t graphics_index = 0;
    for (uint32_t i = 0; i < queue_families.size(); ++i){
        if ((queue_families[i].queueFlags & vk::QueueFlagBits::eGraphics) == vk::QueueFlagBits::eGraphics) {
            graphics_index = i;
            break ;
        }
    }

    vk::PhysicalDeviceProperties pdevice_props = pdevice.getProperties();

    float queue_priorities = { 1.0f };
    vk::DeviceQueueCreateInfo graphics_queue_ci{
            .queueFamilyIndex = graphics_index,
            .queueCount = 1,
            .pQueuePriorities = &queue_priorities,
    };

    vk::PhysicalDeviceFeatures device_features{};

    std::vector<const char*> extensions = { "VK_KHR_swapchain" };
    vk::DeviceCreateInfo device_ci{
            .queueCreateInfoCount = 1,
            .pQueueCreateInfos = &graphics_queue_ci,
            .enabledLayerCount = static_cast<uint32_t>(validation_layers.size()),
            .ppEnabledLayerNames = validation_layers.data(),
            .enabledExtensionCount = static_cast<uint32_t>(extensions.size()),
            .ppEnabledExtensionNames = extensions.data(),
            .pEnabledFeatures = &device_features,
    };

    device = pdevice.createDevice(device_ci);
    vk::Queue graphics_queue = device.getQueue(graphics_index, 0);

    VkSurfaceKHR dummy_surface;
    if (glfwCreateWindowSurface(instance, window, nullptr, &dummy_surface) != VK_SUCCESS) {
        throw std::runtime_error("Surface creation failed");
    }
    vk::SurfaceKHR surface(dummy_surface);

    const vk::Extent2D render_target_size{.width = WIDTH, .height = HEIGHT};

    vk::Format out_format = vk::Format::eB8G8R8A8Unorm;
    vk::ColorSpaceKHR out_color_space = vk::ColorSpaceKHR::eSrgbNonlinear;

    vk::SampleCountFlagBits msaa_samples = get_max_msaa_samples(pdevice_props);
    std::cout << "MSAA_SAMPLES:" << to_string(msaa_samples) << "\n";

    vk::SwapchainCreateInfoKHR swapchain_ci{
            .surface = surface,
            .minImageCount = 3,
            .imageFormat = out_format,
            .imageColorSpace = out_color_space,
            .imageExtent = render_target_size,
            .imageArrayLayers = 1,
            .imageUsage = vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eStorage,
            .imageSharingMode = vk::SharingMode::eExclusive,
            .preTransform = vk::SurfaceTransformFlagBitsKHR::eIdentity,
            .compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque,
            .presentMode = vk::PresentModeKHR::eFifo,
            .clipped = true,
    };

    device.createSwapchainKHR(&swapchain_ci, nullptr, &swapchain);

    vk::AttachmentDescription color_attachment_description{
            .format = out_format,
            .samples = msaa_samples,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
            .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
            .initialLayout = vk::ImageLayout::eUndefined,
            .finalLayout = vk::ImageLayout::eColorAttachmentOptimal
    };
    vk::Format depth_format = vk::Format::eD32Sfloat;
    vk::AttachmentDescription depth_attachment_description{
            .format = depth_format,
            .samples = msaa_samples,
            .loadOp = vk::AttachmentLoadOp::eClear,
            .storeOp = vk::AttachmentStoreOp::eDontCare,
            .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
            .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
            .initialLayout = vk::ImageLayout::eUndefined,
            .finalLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal
    };
    vk::AttachmentDescription resolve_attachment_description{
            .format = out_format,
            .samples = vk::SampleCountFlagBits::e1,
            .loadOp = vk::AttachmentLoadOp::eDontCare,
            .storeOp = vk::AttachmentStoreOp::eStore,
            .stencilLoadOp = vk::AttachmentLoadOp::eDontCare,
            .stencilStoreOp = vk::AttachmentStoreOp::eDontCare,
            .initialLayout = vk::ImageLayout::eUndefined,
            .finalLayout = vk::ImageLayout::ePresentSrcKHR
    };

    vk::AttachmentReference color_attachment_reference{.attachment = 0, .layout = vk::ImageLayout::eColorAttachmentOptimal };
    vk::AttachmentReference depth_attachment_reference{.attachment = 1, .layout = vk::ImageLayout::eDepthStencilAttachmentOptimal };
    vk::AttachmentReference resolve_attachment_reference{.attachment = 2, .layout = vk::ImageLayout::eColorAttachmentOptimal };

    vk::SubpassDescription subpass_descriptions{
            .pipelineBindPoint = vk::PipelineBindPoint::eGraphics,
            .colorAttachmentCount = 1,
            .pColorAttachments = &color_attachment_reference,
            .pResolveAttachments = &resolve_attachment_reference,
            .pDepthStencilAttachment = &depth_attachment_reference,
    };
    vk::SubpassDependency subpass_dependency{
        // The source and destination subpasses are indexed based on their position relative to the
        //pSubpasses structure in the render pass create info
        .srcSubpass = VK_SUBPASS_EXTERNAL,
        .dstSubpass = 0,
        .srcStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests,
        .dstStageMask = vk::PipelineStageFlagBits::eColorAttachmentOutput | vk::PipelineStageFlagBits::eEarlyFragmentTests,
        .srcAccessMask = vk::AccessFlagBits::eNone,
        .dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite | vk::AccessFlagBits::eDepthStencilAttachmentWrite
    };

    std::array<vk::AttachmentDescription, 3> attachments{
        color_attachment_description,
        depth_attachment_description,
        resolve_attachment_description
    };
    vk::RenderPass render_pass = device.createRenderPass({
         .attachmentCount = attachments.size(),
         .pAttachments = attachments.data(),
         .subpassCount = 1,
         .pSubpasses = &subpass_descriptions,
         .dependencyCount = 1,
         .pDependencies = &subpass_dependency
    });

    //Create depth images
    vk::Image depth_image = device.createImage(vk::ImageCreateInfo{
        .imageType = vk::ImageType::e2D,
        .format = depth_format,
        .extent = {WIDTH, HEIGHT, 1},
        .mipLevels = 1,
        .arrayLayers = 1,
        .samples = msaa_samples,
        .tiling = vk::ImageTiling::eOptimal,
        .usage = vk::ImageUsageFlagBits::eDepthStencilAttachment,
        .sharingMode = vk::SharingMode::eExclusive
    });
    vk::MemoryRequirements depth_buffer_memreqs = device.getImageMemoryRequirements(depth_image);
    vk::DeviceMemory depth_buffer_memory = device.allocateMemory(vk::MemoryAllocateInfo{
            .allocationSize = depth_buffer_memreqs.size,
            .memoryTypeIndex = findMemoryType(pdevice, depth_buffer_memreqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal),
    });

    device.bindImageMemory(depth_image, depth_buffer_memory, 0);

    vk::ImageView depth_imageview = device.createImageView(vk::ImageViewCreateInfo{
        .image = depth_image,
        .viewType = vk::ImageViewType::e2D,
        .format = depth_format,
        .components = {
            vk::ComponentSwizzle::eIdentity,
            vk::ComponentSwizzle::eIdentity,
            vk::ComponentSwizzle::eIdentity,
        },
        .subresourceRange = {
            .aspectMask = vk::ImageAspectFlagBits::eDepth,
            .baseMipLevel = 0,
            .levelCount = 1,
            .baseArrayLayer = 0,
            .layerCount = 1,
        },
    });

    //Create depth images
    vk::Image msaa_target = device.createImage(vk::ImageCreateInfo{
            .imageType = vk::ImageType::e2D,
            .format = out_format,
            .extent = {WIDTH, HEIGHT, 1},
            .mipLevels = 1,
            .arrayLayers = 1,
            .samples = msaa_samples,
            .tiling = vk::ImageTiling::eOptimal,
            .usage = vk::ImageUsageFlagBits::eColorAttachment,
            .sharingMode = vk::SharingMode::eExclusive,
    });
    vk::MemoryRequirements msaa_buffer_memreqs = device.getImageMemoryRequirements(msaa_target);
    vk::DeviceMemory msaa_buffer_memory = device.allocateMemory(vk::MemoryAllocateInfo{
            .allocationSize = msaa_buffer_memreqs.size,
            .memoryTypeIndex = findMemoryType(pdevice, msaa_buffer_memreqs.memoryTypeBits, vk::MemoryPropertyFlagBits::eDeviceLocal),
    });
    device.bindImageMemory(msaa_target, msaa_buffer_memory, 0);
    vk::ImageView msaa_imageview = device.createImageView(vk::ImageViewCreateInfo{
            .image = msaa_target,
            .viewType = vk::ImageViewType::e2D,
            .format = out_format,
            .components = {
                vk::ComponentSwizzle::eIdentity,
                vk::ComponentSwizzle::eIdentity,
                vk::ComponentSwizzle::eIdentity,
            },
            .subresourceRange = {
                .aspectMask = vk::ImageAspectFlagBits::eColor,
                .baseMipLevel = 0,
                .levelCount = 1,
                .baseArrayLayer = 0,
                .layerCount = 1,
            },
    });

    std::vector<vk::Image> swapchain_images = device.getSwapchainImagesKHR(swapchain);

    std::vector<vk::ImageView> swapchain_imageviews(swapchain_images.size());
    std::vector<vk::Framebuffer> frame_buffers(swapchain_images.size());

    for(int i = 0; i < swapchain_images.size(); ++i) {
        vk::ImageViewCreateInfo imgview_ci{
                .image = swapchain_images[i],
                .viewType = vk::ImageViewType::e2D,
                .format = out_format,
                .components = {
                        vk::ComponentSwizzle::eIdentity,
                        vk::ComponentSwizzle::eIdentity,
                        vk::ComponentSwizzle::eIdentity,
                        vk::ComponentSwizzle::eIdentity
                },
                .subresourceRange = {
                        .aspectMask = vk::ImageAspectFlagBits::eColor,
                        .baseMipLevel = 0,
                        .levelCount = 1,
                        .baseArrayLayer = 0,
                        .layerCount = 1,
                },
        };

        swapchain_imageviews[i] = device.createImageView(imgview_ci);

        std::array<vk::ImageView, 3> fb_attachments = {
                msaa_imageview,
                depth_imageview,
                swapchain_imageviews[i]
        };
        frame_buffers[i] = device.createFramebuffer(vk::FramebufferCreateInfo{
            .renderPass = render_pass,
            .attachmentCount = fb_attachments.size(),
            .pAttachments = fb_attachments.data(),
            .width = WIDTH,
            .height = HEIGHT,
            .layers = 1
        });

    }

    //Indices for vertices and normals are not the same, so we copy the right index
    std::vector<glm::vec3> object_data;
    object_data.reserve(attrib.vertices.size() / 3 + attrib.normals.size() / 3);
    std::vector<uint32_t> indices_data;
    indices_data.reserve(shapes[0].mesh.indices.size());

    for (int i = 0; (3 * i + 2) < attrib.vertices.size(); i += 1) {
        object_data.emplace_back(glm::vec3{
                attrib.vertices[3 * i + 0],
                -attrib.vertices[3 * i + 1],
                attrib.vertices[3 * i + 2]
        });
        object_data.emplace_back(glm::vec3{0,0,0});
    }
    for (int i = 0; i < shapes[0].mesh.indices.size(); ++i) {
        const tinyobj::index_t index = shapes[0].mesh.indices[i];
        int normal_index = index.normal_index;
        indices_data.emplace_back(index.vertex_index);

        if (attrib.normals.empty()){
            object_data[2 * index.vertex_index + 1] = glm::zero<glm::vec3>();
        }else{
            object_data[2 * index.vertex_index + 1] = glm::vec3{
                attrib.normals[3 * normal_index + 0],
                attrib.normals[3 * normal_index + 1],
                attrib.normals[3 * normal_index + 2]
            };
        }
    }

    size_t data_size = object_data.size() * sizeof(glm::vec3);
    vk::Buffer data_buffer = device.createBuffer(vk::BufferCreateInfo{
            .size = data_size,
            .usage = vk::BufferUsageFlagBits::eVertexBuffer,
            .sharingMode = vk::SharingMode::eExclusive
    });
    size_t index_size = indices_data.size() * sizeof(uint32_t);
    vk::Buffer index_buffer = device.createBuffer(vk::BufferCreateInfo{
            .size = index_size,
            .usage = vk::BufferUsageFlagBits::eIndexBuffer,
            .sharingMode = vk::SharingMode::eExclusive
    });

    vk::MemoryRequirements data_requirements = device.getBufferMemoryRequirements(data_buffer);
    vk::MemoryRequirements index_requirements = device.getBufferMemoryRequirements(index_buffer);

    vk::DeviceMemory object_memory = device.allocateMemory({
        .allocationSize = data_size,
        .memoryTypeIndex = findMemoryType(pdevice, data_requirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent)
    });
    vk::DeviceMemory index_memory = device.allocateMemory({
        .allocationSize = index_size,
        .memoryTypeIndex = findMemoryType(pdevice, index_requirements.memoryTypeBits, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent)
    });

    void *data = device.mapMemory(object_memory, 0, data_size);
    memcpy(data, object_data.data(), data_size);
    device.unmapMemory(object_memory);

    data = device.mapMemory(index_memory, 0, index_size);
    memcpy(data, indices_data.data(), index_size);
    device.unmapMemory(index_memory);
    std::cout << 2 << " ALLOCATIONS OF DEVICE MEMORY" << "\n";

    device.bindBufferMemory(data_buffer, object_memory, 0);
    device.bindBufferMemory(index_buffer, index_memory, 0);

    //Create the shaders uniforms
    //Describes which buffer to take the vertex data from
    vk::VertexInputBindingDescription vertex_input_bindings{
            .binding = 0,
            .stride = 2 * sizeof(glm::vec3),
            .inputRate = vk::VertexInputRate::eVertex
    };

    //Position and normals
    std::vector<vk::VertexInputAttributeDescription> vertex_inputs_attributes{
        vk::VertexInputAttributeDescription{
            .location = 0,
            .binding = 0,
            .format = vk::Format::eR32G32B32Sfloat,
            .offset = 0,
        },
        vk::VertexInputAttributeDescription{
            .location = 1,
            .binding = 0,
            .format = vk::Format::eR32G32B32Sfloat,
            .offset = sizeof(glm::vec3),
        }
    };

    //Create the shaders and the pipelines
    std::ifstream vs_file("vertex.sprv", std::ios::binary);
    std::ifstream fs_file("fragment.sprv", std::ios::binary);
    std::vector<char> vs_code{std::istreambuf_iterator<char>(vs_file), {}};
    std::vector<char> fs_code{std::istreambuf_iterator<char>(fs_file), {}};

    vk::ShaderModule vertex_shader = device.createShaderModule({.codeSize = vs_code.size(), .pCode = reinterpret_cast<const uint32_t*>(vs_code.data())});
    vk::ShaderModule fragment_shader = device.createShaderModule({.codeSize = fs_code.size(), .pCode = reinterpret_cast<const uint32_t*>(fs_code.data())});

    std::array<vk::PipelineShaderStageCreateInfo, 2> shaders_stages{
        vk::PipelineShaderStageCreateInfo{ .stage = vk::ShaderStageFlagBits::eVertex,
                                           .module = vertex_shader, .pName = "main" },
        vk::PipelineShaderStageCreateInfo{ .stage = vk::ShaderStageFlagBits::eFragment,
                                           .module = fragment_shader, .pName = "main" }
    };

    vk::PipelineVertexInputStateCreateInfo vertex_input_ci{
        .vertexBindingDescriptionCount = 1,
        .pVertexBindingDescriptions = &vertex_input_bindings,
        .vertexAttributeDescriptionCount = static_cast<uint32_t>(vertex_inputs_attributes.size()),
        .pVertexAttributeDescriptions = vertex_inputs_attributes.data()
    };
    vk::PipelineInputAssemblyStateCreateInfo vertex_assembly_ci{ .topology = vk::PrimitiveTopology::eTriangleList };
    vk::Viewport viewport{ .x = 0.0f, .y = 0.0, .width = 800.0f, .height = 600.0f, .minDepth = 0.0f, .maxDepth = 1.0f };
    vk::Rect2D scissor{ .offset = {0, 0}, .extent = swapchain_ci.imageExtent };
    vk::PipelineViewportStateCreateInfo pipeline_vw_ci{
        .viewportCount = 1, .pViewports = &viewport,
        .scissorCount = 1, .pScissors = &scissor
    };
    vk::PipelineRasterizationStateCreateInfo rasterizer_ci{
        .polygonMode = vk::PolygonMode::eFill,
        .cullMode = vk::CullModeFlagBits::eBack,
        .frontFace = vk::FrontFace::eCounterClockwise,
        .lineWidth = 1.0f
    };
    vk::PipelineMultisampleStateCreateInfo multisampling_settings{
        .rasterizationSamples = msaa_samples,
    };

    vk::PipelineColorBlendAttachmentState blend_settings_for_attachment{
        .blendEnable = false,
        .colorWriteMask = vk::ColorComponentFlagBits::eR
                | vk::ColorComponentFlagBits::eG
                | vk::ColorComponentFlagBits::eB
                | vk::ColorComponentFlagBits::eA,
    };
    vk::PipelineColorBlendStateCreateInfo global_blend_settings {
        .logicOpEnable = false,
        .attachmentCount = 1,
        .pAttachments = &blend_settings_for_attachment
    };

    vk::PushConstantRange push_constant_range{
        .stageFlags = vk::ShaderStageFlagBits::eVertex,
        .offset = 0,
        .size = 2 * sizeof(glm::mat4),
    };
    vk::PipelineLayout pipeline_layout = device.createPipelineLayout(vk::PipelineLayoutCreateInfo{
        .pushConstantRangeCount = 1,
        .pPushConstantRanges = &push_constant_range
    });

    vk::PipelineDepthStencilStateCreateInfo depth_state_ci{
        .depthTestEnable = true,
        .depthWriteEnable = true,
        .depthCompareOp = vk::CompareOp::eLess,
        .depthBoundsTestEnable = false,
        .stencilTestEnable = false,
        .front = {},
        .back = {},
        .minDepthBounds = 0.0f,
        .maxDepthBounds = 1.0f,
    };

    vk::GraphicsPipelineCreateInfo pipeline_ci{
        .stageCount = shaders_stages.size(),
        .pStages = shaders_stages.data(),
        .pVertexInputState = &vertex_input_ci,
        .pInputAssemblyState = &vertex_assembly_ci,
        .pViewportState = &pipeline_vw_ci,
        .pRasterizationState = &rasterizer_ci,
        .pMultisampleState = &multisampling_settings,
        .pDepthStencilState = &depth_state_ci,
        .pColorBlendState = &global_blend_settings,
        .layout = pipeline_layout,
        .renderPass = render_pass,
        .subpass = 0,
    };

    vk::Pipeline object_pipeline = device.createGraphicsPipeline({}, pipeline_ci).value;

    vk::CommandPool command_pool = device.createCommandPool({
        .flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer,
        .queueFamilyIndex = graphics_index
    });

    std::vector<vk::CommandBuffer> render_command_buffers = device.allocateCommandBuffers({
        .commandPool = command_pool,
        .level = vk::CommandBufferLevel::ePrimary,
        .commandBufferCount = 1
    });

    vk::Semaphore next_image_available  = device.createSemaphore({});
    vk::Semaphore render_finished  = device.createSemaphore({});

    vk::Fence waitForGPU = device.createFence({ .flags = vk::FenceCreateFlagBits::eSignaled });

    uint32_t max_timeout = std::numeric_limits<uint32_t>::max();

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    (void)io;
    io.Fonts->Build();
    ImGui::StyleColorsDark();

    const std::array<vk::DescriptorPoolSize, 11> imgui_pool_sizes{{
            vk::DescriptorPoolSize{vk::DescriptorType::eSampler, 1000},
            vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, 1000},
            vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, 1000},
            vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 1000},
            vk::DescriptorPoolSize{vk::DescriptorType::eUniformTexelBuffer, 1000},
            vk::DescriptorPoolSize{vk::DescriptorType::eStorageTexelBuffer, 1000},
            vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, 1000},
            vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 1000},
            vk::DescriptorPoolSize{vk::DescriptorType::eUniformBufferDynamic, 1000},
            vk::DescriptorPoolSize{vk::DescriptorType::eStorageBufferDynamic, 1000},
            vk::DescriptorPoolSize{vk::DescriptorType::eInputAttachment, 1000}
    }};

    vk::DescriptorPool gui_pool = device.createDescriptorPool(vk::DescriptorPoolCreateInfo{
        .maxSets = 64,
        .poolSizeCount = imgui_pool_sizes.size(),
        .pPoolSizes = imgui_pool_sizes.data(),
    });

    ImGui_ImplGlfw_InitForVulkan(window, true);
    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.Instance = instance;
    init_info.PhysicalDevice = pdevice;
    init_info.DescriptorPool = gui_pool,
    init_info.Device = device;
    init_info.QueueFamily = graphics_index;
    init_info.Queue = graphics_queue;
    init_info.PipelineCache = VK_NULL_HANDLE,
    init_info.CheckVkResultFn = VK_NULL_HANDLE,
    init_info.Allocator = VK_NULL_HANDLE,
    init_info.Subpass = 0;
    init_info.MinImageCount = 3;
    init_info.ImageCount = 3;
    init_info.MSAASamples = VkSampleCountFlagBits(msaa_samples);
    ImGui_ImplVulkan_Init(&init_info, render_pass);

    vk::PipelineStageFlags wait_stages = vk::PipelineStageFlagBits::eColorAttachmentOutput;

    render_command_buffers[0].reset();
    render_command_buffers[0].begin(vk::CommandBufferBeginInfo{});
    ImGui_ImplVulkan_CreateFontsTexture(render_command_buffers[0]);
    render_command_buffers[0].end();
    graphics_queue.submit(vk::SubmitInfo{
            .commandBufferCount = 1,
            .pCommandBuffers = &render_command_buffers[0],
    });
    graphics_queue.waitIdle();
    render_command_buffers[0].reset();

    glm::mat4 model = glm::mat4(1.0f);
    model = glm::translate(model, glm::vec3{0.0f, 0.0f, -5.0f});
    glm::mat4 projection = glm::perspective(
            glm::radians(70.f),
            static_cast<float>(WIDTH) / static_cast<float>(HEIGHT),
            0.1f, 1000.0f);
    while(!glfwWindowShouldClose(window)){
        glfwPollEvents();

        cameraControls(window, model);

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::Begin("Hello, world!");
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
                    1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
        ImGui::End();

        ImGui::Render();
        ImDrawData* ui_draw_data = ImGui::GetDrawData();

        device.waitForFences(waitForGPU, true, max_timeout);
        device.resetFences(waitForGPU);

        uint32_t image_index = device.acquireNextImageKHR(swapchain, max_timeout, next_image_available, nullptr).value;

        std::array<vk::ClearValue, 2> clear_values{};
        clear_values[0].color = vk::ClearColorValue{{{0.15f, 0.15f, 0.15f, 1.0f}}};
        clear_values[1].depthStencil = vk::ClearDepthStencilValue{1.0f, 0};

        render_command_buffers[0].reset();
        render_command_buffers[0].begin(vk::CommandBufferBeginInfo{});
        render_command_buffers[0].beginRenderPass(vk::RenderPassBeginInfo{
                .renderPass = render_pass,
                .framebuffer = frame_buffers[image_index],
                .renderArea = { .offset = {0, 0}, .extent = swapchain_ci.imageExtent },
                .clearValueCount = clear_values.size(),
                .pClearValues = clear_values.data(),
        }, vk::SubpassContents::eInline);

        render_command_buffers[0].bindPipeline(vk::PipelineBindPoint::eGraphics, object_pipeline);

        glm::mat4 render_matrix[2] = {projection, model};
        render_command_buffers[0].pushConstants(pipeline_layout, vk::ShaderStageFlagBits::eVertex, 0, 2 * sizeof(glm::mat4), &render_matrix);

        vk::DeviceSize offsets = { 0 };
        render_command_buffers[0].bindVertexBuffers(0, data_buffer, offsets);
        render_command_buffers[0].bindIndexBuffer(index_buffer,0,vk::IndexType::eUint32);
        render_command_buffers[0].drawIndexed(indices_data.size(), 1, 0, 0, 0);

        ImGui_ImplVulkan_RenderDrawData(ui_draw_data, render_command_buffers[0]);

        render_command_buffers[0].endRenderPass();

        render_command_buffers[0].end();

        graphics_queue.submit(vk::SubmitInfo{
            .waitSemaphoreCount = 1,
            .pWaitSemaphores = &next_image_available,
            .pWaitDstStageMask = &wait_stages,
            .commandBufferCount = 1,
            .pCommandBuffers = &render_command_buffers[0],
            .signalSemaphoreCount = 1,
            .pSignalSemaphores = &render_finished
        }, waitForGPU);

        graphics_queue.presentKHR(vk::PresentInfoKHR{
            .waitSemaphoreCount = 1, .pWaitSemaphores = &render_finished,
            .swapchainCount = 1,
            .pSwapchains = &swapchain,
            .pImageIndices = &image_index
        });

    }

    device.waitIdle();

    glfwDestroyWindow(window);

    glfwTerminate();

    return 0;
}
