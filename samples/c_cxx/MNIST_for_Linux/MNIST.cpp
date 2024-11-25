// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <iostream>
#include <algorithm>
#include <onnxruntime_cxx_api.h>
#include <array>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "test"};

// This is the structure to interface with the MNIST model
// After instantiation, set the input_image_ data to be the 28x28 pixel image of the number to recognize
// Then call Run() to fill in the results_ data with the probabilities of each
// result_ holds the index with highest probability (aka the number the model thinks is in the image)
struct MNIST {
    MNIST() {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        input_tensor_ = Ort::Value::CreateTensor<float>(memory_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());
        output_tensor_ = Ort::Value::CreateTensor<float>(memory_info, results_.data(), results_.size(), output_shape_.data(), output_shape_.size());
    }

    std::ptrdiff_t Run(const std::string& img_path) {
        const char* input_names[] = {"Input3"};
        const char* output_names[] = {"Plus214_Output_0"};

        int img_width = 0;
        int img_height = 0;
        int img_channels = 0;
        unsigned char* img_gray = stbi_load(img_path.c_str(), &img_width, &img_height, &img_channels, 1);
        for (size_t idx = 0; idx < input_image_.size(); ++idx)
        {
            input_image_[idx] = static_cast<float>(img_gray[idx]) / 255.0f;
        }
        session_.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor_, 1, output_names, &output_tensor_, 1);
        stbi_image_free(img_gray);
        result_ = std::distance(results_.begin(), std::max_element(results_.begin(), results_.end()));
        return result_;
    }

    static constexpr const int width_ = 28;
    static constexpr const int height_ = 28;

    std::array<float, width_ * height_> input_image_{};
    std::array<float, 10> results_{};
    int64_t result_{0};

private:
    Ort::Session session_ = {env, "model.onnx", Ort::SessionOptions{nullptr}};

    Ort::Value input_tensor_{nullptr};
    std::array<int64_t, 4> input_shape_{1, 1, width_, height_};

    Ort::Value output_tensor_{nullptr};
    std::array<int64_t, 2> output_shape_{1, 10};
};

int main(int argc, const char* argv[])
{
    if(2 != argc)
    {
        std::cout << "Usage: " << argv[0] << " <image_path>" << std::endl;
        return -1;
    }
    MNIST mnist;
    mnist.Run(argv[1]);
    std::cout << argv[1] << " : " <<mnist.result_ << std::endl;
}