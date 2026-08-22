// Dear ImGui: standalone example application for GLFW + Metal, using programmable pipeline
// (GLFW is a cross-platform general purpose library for handling windows, inputs, OpenGL/Vulkan/Metal graphics context creation, etc.)

// Learn about Dear ImGui:
// - FAQ                  https://dearimgui.com/faq
// - Getting Started      https://dearimgui.com/getting-started
// - Documentation        https://dearimgui.com/docs (same as your local docs/ folder).
// - Introduction, links and more at the top of imgui.cpp

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_metal.h"
#include "implot.h"
#include "implot3d.h"
#include <cstdio>
#include <omp.h>

#define GLFW_INCLUDE_NONE
#define GLFW_EXPOSE_NATIVE_COCOA
#include <GLFW/glfw3.h>
#include <GLFW/glfw3native.h>

#import <Metal/Metal.h>
#import <QuartzCore/QuartzCore.h>

#import "app.h"
#import "theme.h"

static void glfw_error_callback(int error, const char* description)
{
    if (std::fprintf(stderr, "Glfw Error %d: %s\n", error, description) < 0)
    {
        return;
    }
}

int main()
{

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImPlot3D::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;  // Enable Keyboard Controls
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;   // Enable Gamepad Controls
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;

    //io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\segoeui.ttf");
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/DroidSans.ttf");
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/Roboto-Medium.ttf");
    //io.Fonts->AddFontFromFileTTF("../../misc/fonts/Cousine-Regular.ttf");
    //ImFont* font = io.Fonts->AddFontFromFileTTF("c:\\Windows\\Fonts\\ArialUni.ttf");
    //IM_ASSERT(font != nullptr);

    // Setup window
    glfwSetErrorCallback(glfw_error_callback);
    if (glfwInit() == 0)
        return 1;

    // Create window with graphics context
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    const float main_scale = ImGui_ImplGlfw_GetContentScaleForMonitor(glfwGetPrimaryMonitor());
    GLFWwindow* window = glfwCreateWindow((int)(1880 * main_scale), (int)(1400 * main_scale), "FDN Sandbox", nullptr, nullptr);
    if (window == nullptr)
        return 1;

    const id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (device == nil)
    {
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    const id<MTLCommandQueue> commandQueue = [device newCommandQueue];
    if (commandQueue == nil)
    {
        [device release];
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    // Setup Platform/Renderer backends
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplMetal_Init(device);

    NSWindow* const nswin = glfwGetCocoaWindow(window);
    CAMetalLayer* const layer = [CAMetalLayer layer];
    layer.device = device;
    layer.pixelFormat = MTLPixelFormatBGRA8Unorm;
    nswin.contentView.layer = layer;
    nswin.contentView.wantsLayer = YES;

    MTLRenderPassDescriptor* const renderPassDescriptor = [MTLRenderPassDescriptor new];
    if (renderPassDescriptor == nil)
    {
        ImGui_ImplMetal_Shutdown();
        ImGui_ImplGlfw_Shutdown();
        ImPlot3D::DestroyContext();
        ImPlot::DestroyContext();
        ImGui::DestroyContext();
        [commandQueue release];
        [device release];
        glfwDestroyWindow(window);
        glfwTerminate();
        return 1;
    }

    // Our state
    const ImVec4 clear_color = fdn_sandbox::theme::Color(fdn_sandbox::theme::ColorRole::ApplicationBackground);

    // Set openmp threads to 4 because otherwise it might try to use the economy cores
    omp_set_num_threads(4);

    {
        FDNToolboxApp app(main_scale);

        // Main loop
        while (glfwWindowShouldClose(window) == 0)
        {
            @autoreleasepool
            {
                // Poll and handle events (inputs, window resize, etc.)
                // You can read the io.WantCaptureMouse, io.WantCaptureKeyboard flags to tell if dear imgui wants to
                // use your inputs. When either is true, do not dispatch or overwrite the corresponding input state.
                glfwPollEvents();

                int width = 0;
                int height = 0;
                glfwGetFramebufferSize(window, &width, &height);
                layer.drawableSize = CGSizeMake(width, height);
                const id<CAMetalDrawable> drawable = [layer nextDrawable];

                const id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
                renderPassDescriptor.colorAttachments[0].clearColor =
                    MTLClearColorMake(clear_color.x * clear_color.w, clear_color.y * clear_color.w,
                                      clear_color.z * clear_color.w, clear_color.w);
                renderPassDescriptor.colorAttachments[0].texture = drawable.texture;
                renderPassDescriptor.colorAttachments[0].loadAction = MTLLoadActionClear;
                renderPassDescriptor.colorAttachments[0].storeAction = MTLStoreActionStore;
                const id<MTLRenderCommandEncoder> renderEncoder =
                    [commandBuffer renderCommandEncoderWithDescriptor:renderPassDescriptor];
                [renderEncoder pushDebugGroup:@"ImGui demo"];

                // Start the Dear ImGui frame
                ImGui_ImplMetal_NewFrame(renderPassDescriptor);
                ImGui_ImplGlfw_NewFrame();
                ImGui::NewFrame();

                app.loop();

                // Rendering
                ImGui::Render();
                ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(), commandBuffer, renderEncoder);

                [renderEncoder popDebugGroup];
                [renderEncoder endEncoding];

                [commandBuffer presentDrawable:drawable];
                [commandBuffer commit];
            }
        }
    }

    // Cleanup
    ImGui_ImplMetal_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImPlot3D::DestroyContext();
    ImPlot::DestroyContext();
    ImGui::DestroyContext();

    [renderPassDescriptor release];
    [commandQueue release];
    [device release];

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}